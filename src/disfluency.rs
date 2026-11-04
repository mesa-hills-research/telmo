//! Deeper cleanup of English dictation. After the [fillers](crate::fillers) go, a small tagger
//! and a set of rules delete false starts, repairs, repeats and filler words, once over a whole
//! utterance.
//!
//! ```text
//! We need to, we need to fix the the login page first.
//! We need to fix the login page first.
//! ```
//!
//! It only deletes, so it never changes or adds a word, and punctuation and capitals are tidied
//! as for fillers. The tagger is an encoder (ModernBERT, as Ettin's 68M model) trained to tag
//! each word, and may hold several members whose probabilities are averaged. Its pack holds the
//! weights, the tokenizer and the thresholds it was tuned with.
//!
//! Each word of the text, as the speech model wrote it, is deleted when
//!
//! - the filler rules delete it,
//! - the rules of [`rules`](self) delete it (cut-off words, a preposition started again, a
//!   phrase or function word said twice, "sorry" or "or rather" repairs, a sentence-opening
//!   marker such as "so" or "well" before a clause, and a trailing "so" or "you know"), or
//! - the tagger's probability of a deleting tag reaches its group's threshold. A word in a run of
//!   repeated words can have a group of its own, and any other word goes by its most likely tag.
//!
//! Words of a special word or a spoken form's word always stay.
//!
//! [`SpecialWords::set_disfluencies`](crate::SpecialWords::set_disfluencies) turns it on for
//! transcription and streams. A stream then holds each utterance's text as tentative text, with
//! the fillers left out, until the utterance ends, and settles the cleaned text then.

mod bpe;
mod encoder;
mod rules;
mod units;

use std::path::Path;

pub use bpe::ByteLevelTokenizer;
pub use encoder::Config;

use crate::error::{Error, Result};
use crate::fillers::{Edit, Fillers, Protected};
use crate::json;
use crate::kernels::{Isa, Kernels};
use crate::pack::{MemorySink, Pack, Sink};
use crate::parakeet::Options;
use crate::quant::Precision;
use crate::safetensors::SafeTensors;
use crate::source::Source;
use crate::threads::Pool;
use encoder::{Member, Runner};

/// The model name of a tagger's pack.
pub const PACK_MODEL: &str = "disfluency-tagger";

/// The 8-bit precision a tagger's pack ships in. Its deletions equal f32's on 99.6% of the
/// development utterances it was tuned on.
pub const RECOMMENDED_PRECISION: Precision = Precision::Q8_0;

/// How the tagger's probabilities become deletions, from the pack's `tagger.json`.
#[derive(Debug, Clone, PartialEq)]
struct Settings {
    /// The tags, in the order of the classifier's outputs.
    tags: Vec<String>,
    /// The positions of the tags that delete a word.
    delete: Vec<usize>,
    keep: usize,
    members: usize,
    /// Added to the log of the keep tag's averaged probability.
    keep_bias: f32,
    groups: Vec<Group>,
    /// Each group's threshold.
    thresholds: Vec<f64>,
    /// Words before the first word a window decides, when a text needs several.
    left_context: usize,
    max_tokens: usize,
}

/// A set of words that share a threshold.
#[derive(Debug, Clone, PartialEq)]
enum Group {
    /// Words in a run of copies of a span of words.
    Copies,
    /// Words whose most likely deleting tag starts with one of these letters.
    Tags(String),
}

impl Settings {
    fn from_json(bytes: &[u8], config: &Config) -> Result<Settings> {
        let bad = |m: &str| Error::Model(format!("The tagger's settings aren't usable. {m}"));
        let root = json::parse(bytes)?;
        let strings = |key: &str| -> Option<Vec<String>> {
            let list = root.get(key)?.as_array()?;
            list.iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect()
        };
        let tags = strings("tags").ok_or_else(|| bad("It lists no tags."))?;
        let delete_names = strings("delete").ok_or_else(|| bad("It lists no deleting tags."))?;
        let delete: Vec<usize> = (delete_names.iter())
            .map(|d| tags.iter().position(|t| t == d))
            .collect::<Option<_>>()
            .ok_or_else(|| bad("A deleting tag isn't one of the tags."))?;
        let keep = tags
            .iter()
            .position(|t| t == "O")
            .ok_or_else(|| bad("It has no keep tag, O."))?;
        let letters: String = delete
            .iter()
            .filter_map(|&d| tags[d].chars().next())
            .collect();
        let groups: Vec<Group> = match strings("groups") {
            Some(list) => (list.iter())
                .map(|g| match g.strip_prefix("class:") {
                    Some("copies") => Ok(Group::Copies),
                    Some(other) => Err(bad(&format!(
                        "It names a word class Telmo lacks ({other})."
                    ))),
                    None => Ok(Group::Tags(g.clone())),
                })
                .collect::<Result<_>>()?,
            None => vec![Group::Tags(letters.clone())],
        };
        let mut covered: Vec<char> = (groups.iter())
            .filter_map(|g| match g {
                Group::Tags(l) => Some(l.chars()),
                Group::Copies => None,
            })
            .flatten()
            .collect();
        covered.sort_unstable();
        let mut want: Vec<char> = letters.chars().collect();
        want.sort_unstable();
        if covered != want || letters.len() != delete.len() {
            return Err(bad("Its groups don't cover each deleting tag once."));
        }
        let number = |v: &json::Value| v.as_f64();
        let thresholds: Vec<f64> = match root.get("thresholds").and_then(|t| t.as_array()) {
            Some(list) => list.iter().map(number).collect::<Option<_>>(),
            None => root.get("threshold").and_then(number).map(|t| vec![t]),
        }
        .ok_or_else(|| bad("It has no thresholds."))?;
        if thresholds.len() != groups.len() {
            return Err(bad("Its thresholds don't match its groups."));
        }
        let count = |key: &str, default: usize| {
            root.get(key)
                .map_or(Some(default as u64), |v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| bad(&format!("Its {key} isn't a count.")))
        };
        let settings = Settings {
            members: count("members", 1)?,
            keep_bias: root.get("keep_bias").and_then(number).unwrap_or(0.0) as f32,
            left_context: count("left_context_words", 60)?,
            max_tokens: count("max_tokens", 128)?,
            tags,
            delete,
            keep,
            groups,
            thresholds,
        };
        if settings.tags.len() != config.labels
            || !(1..=16).contains(&settings.members)
            || !(3..=8192).contains(&settings.max_tokens)
        {
            return Err(bad("Its tags or sizes don't fit the model."));
        }
        Ok(settings)
    }

    /// The group of a word with the most likely deleting tag `tag`, in a run of copies or not.
    fn group(&self, tag: usize, in_copies: bool) -> usize {
        let letter = self.tags[tag].chars().next();
        (self.groups.iter())
            .position(|g| match g {
                Group::Copies => in_copies,
                Group::Tags(letters) => letter.is_some_and(|l| letters.contains(l)),
            })
            .expect("the groups cover every deleting tag")
    }
}

/// What the cleanup decides for one word.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    /// Whether the word goes.
    pub deleted: bool,
    /// Whether the filler rules delete it.
    pub filler: bool,
    /// Whether the other rules delete it.
    pub rule: bool,
    /// The tagger's probability that the word is a disfluency to delete.
    pub probability: f32,
    /// The first letter of its most likely deleting tag, such as `R` for a reparandum.
    pub tag: char,
}

/// A loaded disfluency tagger, with its rules: the light one, whose rules are built in, or one
/// with a trained model from a pack.
pub struct Tagger {
    model: Option<Model>,
    fillers: Fillers,
}

/// A trained tagger's weights, tokenizer and settings.
struct Model {
    config: Config,
    settings: Settings,
    tokenizer: ByteLevelTokenizer,
    members: Vec<Member>,
    kernels: Kernels,
    pool: Pool,
    precision: Option<Precision>,
}

impl std::fmt::Debug for Tagger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tagger")
            .field("members", &self.members())
            .field("precision", &self.precision())
            .finish_non_exhaustive()
    }
}

/// Converts a tagger's checkpoint into pack entries in `precision`. `src` holds the weights
/// under Hugging Face's names, each member's after `members.N.` when there are several, and
/// `tagger_json` the settings. The matrices and the token embedding take `precision`, which
/// for 8-bit gives each embedding row its own scale (int8) or one per 32 values (q8_0).
pub fn convert(
    src: &mut dyn Source,
    config_json: &[u8],
    tokenizer_json: &[u8],
    tagger_json: &[u8],
    precision: Precision,
    sink: &mut dyn Sink,
) -> Result<()> {
    let config = Config::from_json(config_json)?;
    let settings = Settings::from_json(tagger_json, &config)?;
    check_tokenizer(&ByteLevelTokenizer::from_json(tokenizer_json)?, &config)?;
    sink.bytes("config.json", config_json)?;
    sink.bytes("tokenizer.json", tokenizer_json)?;
    sink.bytes("tagger.json", tagger_json)?;
    for m in 0..settings.members {
        let from = if settings.members > 1 {
            format!("members.{m}.")
        } else {
            String::new()
        };
        encoder::convert_member(
            src,
            &config,
            &from,
            &format!("members.{m}."),
            precision,
            sink,
        )?;
    }
    Ok(())
}

/// Converts a checkpoint directory (`model.safetensors`, `config.json`, `tokenizer.json` and
/// `tagger.json`) into pack entries.
pub fn convert_path(path: &Path, precision: Precision, sink: &mut dyn Sink) -> Result<()> {
    let read = |name: &str| {
        let p = path.join(name);
        std::fs::read(&p).map_err(|e| Error::io(&p, e))
    };
    let (config, tokenizer, tagger) = (
        read("config.json")?,
        read("tokenizer.json")?,
        read("tagger.json")?,
    );
    let mut st = SafeTensors::open(&path.join("model.safetensors"))?;
    convert(&mut st, &config, &tokenizer, &tagger, precision, sink)
}

fn check_tokenizer(tokenizer: &ByteLevelTokenizer, config: &Config) -> Result<()> {
    if tokenizer.len() > config.vocab
        || config.cls as usize >= config.vocab
        || config.sep as usize >= config.vocab
    {
        return Err(Error::Model(format!(
            "The tokenizer has {} tokens but the model {}.",
            tokenizer.len(),
            config.vocab
        )));
    }
    Ok(())
}

impl Tagger {
    /// The light tagger, a network of hand-set units over each word's spelling, its word class
    /// and the next three words, whose rules are built in. Beside the fillers, it deletes cut-off
    /// words and fragments ("the mi mic microphone"), repeats of a word or a two-word phrase, also
    /// across a hesitation or fragment ("who w who"), and sentence-opening and trailing markers
    /// ("So I think", "I don't have a pen so").
    pub fn light() -> Tagger {
        Tagger {
            model: None,
            fillers: Fillers::for_language("en").expect("English has filler rules"),
        }
    }

    /// Loads a tagger from a pack file or a checkpoint directory, which is converted in memory
    /// to `options.precision`.
    pub fn load(path: &Path, options: &Options) -> Result<Tagger> {
        if path.is_file() {
            return Tagger::from_pack(&Pack::open(path)?, options);
        }
        let mut sink = MemorySink::new(
            PACK_MODEL,
            crate::pack::metadata_for(path, options.precision),
        );
        convert_path(path, options.precision, &mut sink)?;
        Tagger::from_pack(&sink.finish()?, options)
    }

    /// Builds the tagger from a pack's entries, which it keeps views of.
    pub fn from_pack(pack: &Pack, options: &Options) -> Result<Tagger> {
        if pack.model() != PACK_MODEL {
            return Err(Error::Model(format!(
                "The pack holds {}, not a disfluency tagger.",
                pack.model()
            )));
        }
        let isa = options.isa.unwrap_or_else(Isa::detect);
        let kernels = Kernels::new(isa).ok_or_else(|| {
            Error::Model(format!("This CPU can't run the {} kernels.", isa.name()))
        })?;
        let config = Config::from_json(pack.bytes("config.json")?)?;
        let settings = Settings::from_json(pack.bytes("tagger.json")?, &config)?;
        let tokenizer = ByteLevelTokenizer::from_json(pack.bytes("tokenizer.json")?)?;
        check_tokenizer(&tokenizer, &config)?;
        let members = (0..settings.members)
            .map(|m| Member::from_pack(pack, &config, &format!("members.{m}.")))
            .collect::<Result<_>>()?;
        let model = Model {
            config,
            settings,
            tokenizer,
            members,
            kernels,
            pool: Pool::new(options.threads),
            precision: pack.precision(),
        };
        Ok(Tagger {
            model: Some(model),
            ..Tagger::light()
        })
    }

    /// `text` with its fillers and disfluencies left out, its words separated by single
    /// spaces.
    pub fn remove(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        Fillers::text(&words, &self.plan(&words, &Protected::default()))
    }

    /// What happens to each of `words`, a whole utterance as the speech model wrote it.
    pub fn decide(&self, words: &[&str]) -> Vec<Decision> {
        self.decide_protected(words, &Protected::default())
    }

    /// The edits of `words`, a whole utterance, with the words of a `protected` spelling kept.
    pub(crate) fn plan(&self, words: &[&str], protected: &Protected) -> Vec<Edit> {
        let deleted: Vec<bool> = (self.decide_protected(words, protected).iter())
            .map(|d| d.deleted)
            .collect();
        self.fillers.tidy_deletions(words, &deleted, protected)
    }

    fn decide_protected(&self, words: &[&str], protected: &Protected) -> Vec<Decision> {
        let filler = self.fillers.deletions(words, protected);
        let kept = protected.words(&[], words);
        let Some(model) = &self.model else {
            return (units::light(words).into_iter().enumerate())
                .map(|(i, kind)| {
                    let rule = kind.is_some() && !kept[i];
                    Decision {
                        deleted: filler[i] || rule,
                        filler: filler[i],
                        rule,
                        probability: if kind.is_some() { 1.0 } else { 0.0 },
                        tag: kind.and_then(|k| k.chars().next()).unwrap_or('O'),
                    }
                })
                .collect();
        };
        let rule = rules::deletions(words);
        let copies = rules::copies(words);
        let s = &model.settings;
        (model.scores(words).into_iter().enumerate())
            .map(|(i, (probability, tag))| {
                let group = s.group(tag, copies.contains(&i));
                let by_model = probability as f64 >= s.thresholds[group];
                let rule = rule[i] && !kept[i];
                Decision {
                    deleted: filler[i] || (!kept[i] && (rule || by_model)),
                    filler: filler[i],
                    rule,
                    probability,
                    tag: s.tags[tag].chars().next().unwrap_or('?'),
                }
            })
            .collect()
    }

    /// The token ids of one window over `words` from the first, between the classifier's start
    /// and end tokens, and where each word that fits starts, or none for the light tagger.
    /// Words after the first take a space before them, and the window ends before the first
    /// word that doesn't fit.
    pub fn token_ids(&self, words: &[&str]) -> Option<(Vec<u32>, Vec<usize>)> {
        Some(self.model.as_ref()?.token_ids(words))
    }

    /// Member `member`'s logits for every token of `ids`, `[ids.len(), tags]`, or none for the
    /// light tagger. `tap` sees the embeddings' output (`"embeddings"`), each layer's
    /// (`"layer.0"`, ...), the final norm's (`"final_norm"`) and the head's before the
    /// classifier (`"head"`), each `[tokens, hidden]`.
    pub fn logits(
        &self,
        member: usize,
        ids: &[u32],
        tap: &mut dyn FnMut(&str, &[f32]),
    ) -> Option<Vec<f32>> {
        Some(self.model.as_ref()?.logits(member, ids, tap))
    }

    /// The model's architecture, or none for the light tagger.
    pub fn config(&self) -> Option<&Config> {
        self.model.as_ref().map(|m| &m.config)
    }

    /// The members whose probabilities are averaged, none for the light tagger.
    pub fn members(&self) -> usize {
        self.model.as_ref().map_or(0, |m| m.members.len())
    }

    /// The precision of the weight matrices, if they share one.
    pub fn precision(&self) -> Option<Precision> {
        self.model.as_ref().and_then(|m| m.precision)
    }

    pub fn isa(&self) -> Option<Isa> {
        self.model.as_ref().map(|m| m.kernels.isa())
    }

    pub fn threads(&self) -> usize {
        self.model.as_ref().map_or(1, |m| m.pool.threads())
    }

    /// Bytes of weights held in memory.
    pub fn weight_bytes(&self) -> usize {
        self.model
            .as_ref()
            .map_or(0, |m| m.members.iter().map(Member::weight_bytes).sum())
    }
}

impl Model {
    /// Each word's probability of a deleting tag and its most likely deleting tag. A text too
    /// long for one window is read in several, each starting `left_context` words before the
    /// first word it decides.
    fn scores(&self, words: &[&str]) -> Vec<(f32, usize)> {
        let mut out = Vec::with_capacity(words.len());
        while out.len() < words.len() {
            let done = out.len();
            let mut start = done.saturating_sub(self.settings.left_context);
            let (mut ids, mut positions) = self.token_ids(&words[start..]);
            if positions.len() <= done - start {
                // The context leaves no room for the next word, so it's read without any.
                start = done;
                (ids, positions) = self.token_ids(&words[start..]);
            }
            if positions.is_empty() {
                // A word longer than a window is left to the rules.
                out.push((0.0, self.settings.delete[0]));
                continue;
            }
            out.extend(self.window_scores(&ids, &positions[done - start..]));
        }
        out
    }
    /// The scores of the words whose first tokens are at `positions` of `ids`.
    fn window_scores(&self, ids: &[u32], positions: &[usize]) -> Vec<(f32, usize)> {
        let s = &self.settings;
        let labels = self.config.labels;
        let mut mean = vec![0f32; positions.len() * labels];
        for member in 0..self.members.len() {
            let logits = self.logits(member, ids, &mut |_, _| {});
            for (sum, &p) in mean.chunks_exact_mut(labels).zip(positions) {
                let probs = softmax(&logits[p * labels..(p + 1) * labels]);
                for (m, q) in sum.iter_mut().zip(probs) {
                    *m += q;
                }
            }
        }
        let n = self.members.len() as f32;
        mean.chunks_exact(labels)
            .map(|sum| {
                // The members' average, with the keep bias added to its log, renormalized.
                let mut logits: Vec<f32> = sum
                    .iter()
                    .map(|&m| (m / n).max(f32::MIN_POSITIVE).ln())
                    .collect();
                logits[s.keep] += s.keep_bias;
                let probs = softmax(&logits);
                let probability = s.delete.iter().map(|&d| probs[d]).fold(0.0, |a, b| a + b);
                let tag = (s.delete.iter().copied())
                    .fold(None, |best: Option<usize>, d| match best {
                        Some(b) if probs[b] >= probs[d] => Some(b),
                        _ => Some(d),
                    })
                    .expect("a deleting tag");
                (probability, tag)
            })
            .collect()
    }

    fn token_ids(&self, words: &[&str]) -> (Vec<u32>, Vec<usize>) {
        let mut ids = vec![self.config.cls];
        let mut positions = Vec::new();
        for (i, word) in words.iter().enumerate() {
            let piece = if i == 0 {
                self.tokenizer.encode(word)
            } else {
                self.tokenizer.encode(&format!(" {word}"))
            };
            if ids.len() + piece.len() + 1 > self.settings.max_tokens {
                break;
            }
            positions.push(ids.len());
            ids.extend(piece);
        }
        ids.push(self.config.sep);
        (ids, positions)
    }

    fn logits(&self, member: usize, ids: &[u32], tap: &mut dyn FnMut(&str, &[f32])) -> Vec<f32> {
        let runner = Runner {
            config: &self.config,
            kernels: &self.kernels,
            pool: &self.pool,
        };
        runner.logits(&self.members[member], ids, tap)
    }
}

/// Softmax in f32, as the reference computes it.
fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let exp: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;

    /// Random tensors under Hugging Face's names.
    struct Random {
        seed: u64,
        shapes: HashMap<String, Vec<usize>>,
    }

    impl Source for Random {
        fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
            self.shapes.insert(name.to_string(), shape.to_vec());
            let len: usize = shape.iter().product();
            let norm = name.ends_with("norm.weight");
            Ok((0..len)
                .map(|_| {
                    self.seed ^= self.seed << 13;
                    self.seed ^= self.seed >> 7;
                    self.seed ^= self.seed << 17;
                    let r = (self.seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                    if norm { 1.0 + 0.2 * r } else { 0.4 * r }
                })
                .collect())
        }
    }

    /// A byte-level vocabulary of the 256 bytes alone, with `[CLS]` and `[SEP]` after them.
    fn byte_tokenizer() -> String {
        let mut vocab: Vec<(char, u8)> = bpe::byte_of_char().into_iter().collect();
        vocab.sort_by_key(|&(_, b)| b);
        let pieces: Vec<String> = (vocab.iter())
            .map(|&(c, b)| format!("{}:{b}", json::Value::String(c.to_string()).to_json()))
            .collect();
        format!(
            r#"{{"model":{{"type":"BPE","vocab":{{{}}},"merges":[]}},
              "pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":false}},
              "added_tokens":[{{"id":256,"content":"[CLS]","special":true}},
                {{"id":257,"content":"[SEP]","special":true}}]}}"#,
            pieces.join(",")
        )
    }

    /// A tagger of two members with random weights, three layers and a band of two tokens
    /// either side, whose groups all take `threshold`, at most `max_tokens` a window.
    pub(crate) fn tiny(threshold: f64, max_tokens: usize, precision: Precision) -> Arc<Tagger> {
        let labels: Vec<String> = (0..6).map(|i| format!("\"{i}\":\"T{i}\"")).collect();
        let config = format!(
            r#"{{"model_type":"modernbert","attention_bias":false,"mlp_bias":false,
              "norm_bias":false,"classifier_bias":false,"hidden_activation":"gelu",
              "classifier_activation":"gelu","hidden_size":32,"num_attention_heads":2,
              "num_hidden_layers":3,"intermediate_size":32,"vocab_size":258,
              "global_attn_every_n_layers":3,"local_attention":4,"cls_token_id":256,
              "sep_token_id":257,"norm_eps":1e-5,"id2label":{{{}}},
              "rope_parameters":{{"full_attention":{{"rope_theta":160000.0}},
                "sliding_attention":{{"rope_theta":10000.0}}}}}}"#,
            labels.join(",")
        );
        let settings = format!(
            r#"{{"tags":["O","H","F","RM","IM","RP"],"delete":["H","F","RM","IM"],"members":2,
              "keep_bias":0.5,"groups":["class:copies","HFI","R"],
              "thresholds":[{threshold},{threshold},{threshold}],"left_context_words":3,
              "max_tokens":{max_tokens}}}"#
        );
        let mut src = Random {
            seed: 7,
            shapes: HashMap::new(),
        };
        let mut sink = MemorySink::new(PACK_MODEL, Vec::new());
        convert(
            &mut src,
            config.as_bytes(),
            byte_tokenizer().as_bytes(),
            settings.as_bytes(),
            precision,
            &mut sink,
        )
        .unwrap();
        assert!(src.shapes.keys().all(|k| k.starts_with("members.")));
        let options = Options {
            threads: 2,
            ..Options::default()
        };
        Arc::new(Tagger::from_pack(&sink.finish().unwrap(), &options).unwrap())
    }

    #[test]
    fn the_thresholds_decide_what_the_tagger_deletes_beside_the_rules() {
        let text = "Um, so I think the the plan works";
        // Above any probability, only the filler rules and the other rules delete.
        assert_eq!(
            tiny(2.0, 128, Precision::F32).remove(text),
            "I think the plan works"
        );
        assert_eq!(tiny(0.0, 128, Precision::Int8).remove(text), "");
        let decisions = tiny(2.0, 128, Precision::F32).decide(&text.split(' ').collect::<Vec<_>>());
        let flags: Vec<(bool, bool)> = decisions.iter().map(|d| (d.filler, d.rule)).collect();
        assert_eq!(
            flags[..5],
            [
                (true, false),
                (false, true),
                (false, false),
                (false, false),
                (true, false)
            ]
        );
        assert!(
            decisions
                .iter()
                .all(|d| (0.0..=1.0).contains(&d.probability))
        );
        assert!(decisions.iter().all(|d| "HFRI".contains(d.tag)));
    }

    #[test]
    fn the_light_tagger_needs_no_model() {
        let light = Tagger::light();
        assert_eq!((light.members(), light.config().is_none()), (0, true));
        let words: Vec<&str> = "Um the des design is small".split(' ').collect();
        let gone: Vec<bool> = light.decide(&words).iter().map(|d| d.deleted).collect();
        assert_eq!(gone, [true, false, true, false, false, false]);
        assert_eq!(
            light.remove("Um the des design is small"),
            "The design is small"
        );
    }

    #[test]
    fn a_long_text_is_read_in_windows_with_context_before_each() {
        let tagger = tiny(2.0, 12, Precision::F32);
        let words: Vec<&str> = "one two three four five six seven eight nine ten eleven"
            .split(' ')
            .collect();
        let (ids, positions) = tagger.token_ids(&words).unwrap();
        assert!(ids.len() <= 12 && positions.len() < words.len());
        let whole = tagger.decide(&words);
        assert_eq!(whole.len(), words.len());
        // The first window's words score as they do alone.
        let first = tagger.decide(&words[..positions.len()]);
        for (a, b) in first.iter().zip(&whole) {
            assert_eq!(a.probability, b.probability);
        }
        // A word too long for any window is left to the rules.
        let long = "x".repeat(40);
        let decided = tagger.decide(&["so", &long, "go"]);
        assert_eq!((decided.len(), decided[1].probability), (3, 0.0));
    }
}

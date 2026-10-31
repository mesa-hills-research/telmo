//! Downloads files over HTTPS, resuming what an earlier attempt left and checking each against
//! its SHA-256.

use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::sha256;

/// How long a connection may send nothing before the download gives up.
pub const STALL: Duration = Duration::from_secs(30);

/// A client that names Telmo and its version, follows redirects, uses a proxy from the
/// environment (`HTTPS_PROXY` and the like) and gives up on a connection that sends nothing for
/// `stall`.
pub fn agent(stall: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .user_agent(concat!("telmo/", env!("CARGO_PKG_VERSION")))
        .timeout_connect(Some(stall))
        .timeout_per_read(Some(stall))
        .timeout_per_write(Some(stall))
        .http_status_as_error(false)
        .build()
        .into()
}

#[derive(Debug, PartialEq)]
pub enum Error {
    /// The download failed. `retry` says whether running it again may get past it, as after a
    /// dropped connection, and resume it.
    Failed { reason: String, retry: bool },
    /// The file couldn't be read or written, as the message says.
    File(String),
    /// The file didn't match its SHA-256 and was deleted.
    Checksum,
}

/// Makes `path` the file at `url`, `size` bytes with the SHA-256 `sha256` in hex. A shorter file
/// is resumed from where it ends, and a file of the right size is only checked. A file that
/// doesn't match is deleted. Progress shows while it downloads when stderr is a terminal.
pub fn fetch(
    agent: &ureq::Agent,
    url: &str,
    path: &Path,
    size: u64,
    sha256: &str,
) -> Result<(), Error> {
    if fs::metadata(path).map_or(0, |m| m.len()) != size {
        download(agent, url, path, size)?;
    }
    let digest = sha256::file_hex(path)
        .map_err(|e| Error::File(format!("Couldn't read {} ({e}).", path.display())))?;
    if digest != sha256 {
        let _ = fs::remove_file(path);
        return Err(Error::Checksum);
    }
    Ok(())
}

fn download(agent: &ureq::Agent, url: &str, path: &Path, size: u64) -> Result<(), Error> {
    let name = path.file_name().unwrap_or(path.as_os_str()).display();
    match megabytes(size) {
        0 => eprintln!("{name}"),
        mb => eprintln!("{name} ({mb} MB)"),
    }
    // A file longer than it should be can't be resumed.
    let have = fs::metadata(path).map_or(0, |m| m.len());
    let resume_at = if have < size { have } else { 0 };
    let mut request = agent.get(url);
    if resume_at > 0 {
        request = request.header("Range", format!("bytes={resume_at}-"));
    }
    let response = request.call().map_err(failed)?;
    let status = response.status();
    let start = match status.as_u16() {
        // The whole file, also from a server that ignores Range.
        200 => 0,
        206 if range_start(&response) == Some(resume_at) => resume_at,
        206 => {
            return Err(Error::Failed {
                reason: "the server sent a part that wasn't asked for".into(),
                retry: false,
            });
        }
        code => {
            return Err(Error::Failed {
                reason: format!(
                    "the server answered {code} {}",
                    status.canonical_reason().unwrap_or("")
                )
                .trim_end()
                .into(),
                retry: matches!(code, 408 | 429 | 500..),
            });
        }
    };
    let write_error =
        |e: io::Error| Error::File(format!("Couldn't write {} ({e}).", path.display()));
    let mut file = if start == 0 {
        File::create(path)
    } else {
        OpenOptions::new().append(true).open(path)
    }
    .map_err(write_error)?;

    let mut body = response.into_body().into_reader().take(size - start);
    let mut progress = Progress::new(start, size);
    let mut buffer = vec![0; 1 << 16];
    let mut done = start;
    let result = loop {
        match body.read(&mut buffer) {
            Ok(0) if done < size => {
                break Err(Error::Failed {
                    reason: "the connection closed early".into(),
                    retry: true,
                });
            }
            Ok(0) => break Ok(()),
            Ok(n) => {
                if let Err(e) = file.write_all(&buffer[..n]) {
                    break Err(write_error(e));
                }
                done += n as u64;
                progress.show(done);
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => break Err(failed(e.into())),
        }
    };
    progress.finish(done);
    result
}

/// The first byte of a 206 response's `Content-Range: bytes FIRST-LAST/SIZE`.
fn range_start(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let range = response.headers().get("Content-Range")?.to_str().ok()?;
    let (first, _) = range.strip_prefix("bytes ")?.split_once('-')?;
    first.parse().ok()
}

/// What went wrong in a request, in words for a message, and whether running it again may help.
fn failed(e: ureq::Error) -> Error {
    use io::ErrorKind::*;
    use ureq::Error::*;
    let reason = match &e {
        Io(io) => match io.kind() {
            UnexpectedEof => "the connection closed early".into(),
            TimedOut => "timed out".into(),
            ConnectionRefused | HostUnreachable | NetworkUnreachable | AddrNotAvailable => {
                "couldn't reach the server".into()
            }
            _ => io.to_string(),
        },
        Timeout(_) | BodyStalled => "timed out".into(),
        HostNotFound => "couldn't find the server".into(),
        ConnectionFailed => "couldn't reach the server".into(),
        e => e.to_string(),
    };
    let retry = match &e {
        // rustls rejecting the server's certificate comes as InvalidData.
        Io(io) => io.kind() != InvalidData,
        Timeout(_) | BodyStalled | HostNotFound | ConnectionFailed | ConnectProxyFailed(_) => true,
        _ => false,
    };
    Error::Failed { reason, retry }
}

pub fn megabytes(bytes: u64) -> u64 {
    (bytes + 500_000) / 1_000_000
}

/// A line on a terminal that shows how much of a file has arrived and how fast.
struct Progress {
    /// Whether to show it: stderr is a terminal and the file is at least a megabyte.
    on: bool,
    from: u64,
    size: u64,
    started: Instant,
    shown: Instant,
}

impl Progress {
    fn new(from: u64, size: u64) -> Progress {
        Progress {
            on: io::stderr().is_terminal() && megabytes(size) > 0,
            from,
            size,
            started: Instant::now(),
            shown: Instant::now(),
        }
    }

    /// Updates the line a few times a second.
    fn show(&mut self, done: u64) {
        if self.on && self.shown.elapsed() >= Duration::from_millis(250) {
            self.print(done);
        }
    }

    fn print(&mut self, done: u64) {
        self.shown = Instant::now();
        let seconds = self.started.elapsed().as_secs_f64();
        let line = format!(
            "{} of {} MB, {:.1} MB/s",
            megabytes(done),
            megabytes(self.size),
            (done - self.from) as f64 / 1e6 / seconds.max(1e-3)
        );
        // Spaces cover what's left of a longer line before.
        eprint!("\r{line:<30}");
    }

    fn finish(&mut self, done: u64) {
        if self.on {
            self.print(done);
            eprintln!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufRead;
    use std::net::{TcpListener, TcpStream};
    use std::thread;

    /// What the test server does with each request it gets, in order.
    enum Reply {
        /// Sends the whole body, or the part a Range header asks for.
        Body,
        /// Sends the whole body and ignores a Range header.
        IgnoreRange,
        /// Sends the headers for the whole body, then `n` bytes, then closes the connection.
        DropAfter(usize),
        /// Sends the headers and `n` bytes, then nothing while the connection stays open.
        StallAfter(usize),
        /// Redirects to the same path with `?moved` appended.
        Redirect,
        /// Answers with this status and no body.
        Status(u16),
    }

    /// A local HTTP server for `body`, which answers its requests as `replies` say and records
    /// each request's Range header.
    struct Server {
        url: String,
        ranges: thread::JoinHandle<Vec<Option<String>>>,
    }

    impl Server {
        fn start(body: Vec<u8>, replies: Vec<Reply>) -> Server {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("http://{}/model.bin", listener.local_addr().unwrap());
            let ranges = thread::spawn(move || {
                let mut ranges = Vec::new();
                for reply in replies {
                    let (stream, _) = listener.accept().unwrap();
                    ranges.push(answer(stream, &body, reply));
                }
                ranges
            });
            Server { url, ranges }
        }

        /// The Range header of each request the server answered.
        fn ranges(self) -> Vec<Option<String>> {
            self.ranges.join().unwrap()
        }
    }

    fn answer(mut stream: TcpStream, body: &[u8], reply: Reply) -> Option<String> {
        let mut reader = io::BufReader::new(stream.try_clone().unwrap());
        let mut request_line = String::new();
        reader.read_line(&mut request_line).unwrap();
        let path = request_line.split(' ').nth(1).unwrap().to_string();
        let mut range = None;
        let mut user_agent = None;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            let (name, value) = line.split_once(": ").unwrap();
            match name.to_ascii_lowercase().as_str() {
                "range" => range = Some(value.to_string()),
                "user-agent" => user_agent = Some(value.to_string()),
                _ => {}
            }
        }
        assert_eq!(
            user_agent.as_deref(),
            Some(concat!("telmo/", env!("CARGO_PKG_VERSION")))
        );
        let first = range
            .as_deref()
            .and_then(|r| r.strip_prefix("bytes=")?.strip_suffix('-')?.parse().ok());
        let head = |status: &str, part: &[u8], extra: &str| {
            format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
                part.len()
            )
        };
        let mut send = |bytes: &[u8]| {
            // The client may hang up first, as when it gives up on a stall.
            let _ = stream.write_all(bytes);
        };
        match reply {
            Reply::Body => match first {
                Some(first) => {
                    let part = &body[first..];
                    let range = format!(
                        "Content-Range: bytes {first}-{}/{}\r\n",
                        body.len() - 1,
                        body.len()
                    );
                    send(head("206 Partial Content", part, &range).as_bytes());
                    send(part);
                }
                None => {
                    send(head("200 OK", body, "").as_bytes());
                    send(body);
                }
            },
            Reply::IgnoreRange => {
                send(head("200 OK", body, "").as_bytes());
                send(body);
            }
            Reply::DropAfter(n) => {
                send(head("200 OK", body, "").as_bytes());
                send(&body[..n]);
            }
            Reply::StallAfter(n) => {
                send(head("200 OK", body, "").as_bytes());
                send(&body[..n]);
                thread::sleep(Duration::from_secs(3));
            }
            Reply::Redirect => {
                let location = format!("Location: {path}?moved\r\n");
                send(head("302 Found", &[], &location).as_bytes());
            }
            Reply::Status(code) => send(head(&format!("{code} Nope"), &[], "").as_bytes()),
        }
        range
    }

    /// Bytes that differ along the file, so a part in the wrong place changes the digest.
    fn body() -> Vec<u8> {
        (0..300_000u32).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    fn digest(bytes: &[u8]) -> String {
        let mut sha = sha256::Sha256::new();
        sha.update(bytes);
        sha256::hex(&sha.finish())
    }

    fn fetch_from(server: &Server, path: &Path, body: &[u8]) -> Result<(), Error> {
        let agent = agent(Duration::from_secs(1));
        fetch(&agent, &server.url, path, body.len() as u64, &digest(body))
    }

    #[test]
    fn downloads_a_whole_file_through_a_redirect() {
        let dir = crate::tests::temp_dir("download-whole");
        let path = dir.join("model.bin");
        let body = body();
        let server = Server::start(body.clone(), vec![Reply::Redirect, Reply::Body]);
        assert_eq!(fetch_from(&server, &path, &body), Ok(()));
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(server.ranges(), [None, None]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn resumes_after_a_dropped_connection_across_a_redirect() {
        let dir = crate::tests::temp_dir("download-resume");
        let path = dir.join("model.bin");
        let body = body();
        let server = Server::start(
            body.clone(),
            vec![Reply::DropAfter(100_000), Reply::Redirect, Reply::Body],
        );
        let dropped = Error::Failed {
            reason: "the connection closed early".into(),
            retry: true,
        };
        assert_eq!(fetch_from(&server, &path, &body), Err(dropped));
        assert_eq!(fs::read(&path).unwrap(), &body[..100_000], "kept to resume");
        assert_eq!(fetch_from(&server, &path, &body), Ok(()));
        assert_eq!(fs::read(&path).unwrap(), body);
        let resume = Some("bytes=100000-".to_string());
        assert_eq!(server.ranges(), [None, resume.clone(), resume]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn starts_over_when_the_server_ignores_the_range() {
        let dir = crate::tests::temp_dir("download-ignore-range");
        let path = dir.join("model.bin");
        let body = body();
        fs::write(&path, &body[..1000]).unwrap();
        let server = Server::start(body.clone(), vec![Reply::IgnoreRange]);
        assert_eq!(fetch_from(&server, &path, &body), Ok(()));
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(server.ranges(), [Some("bytes=1000-".to_string())]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn starts_over_when_the_file_is_too_long() {
        let dir = crate::tests::temp_dir("download-too-long");
        let path = dir.join("model.bin");
        let body = body();
        fs::write(&path, [body.as_slice(), b"extra"].concat()).unwrap();
        let server = Server::start(body.clone(), vec![Reply::Body]);
        assert_eq!(fetch_from(&server, &path, &body), Ok(()));
        assert_eq!(fs::read(&path).unwrap(), body);
        assert_eq!(server.ranges(), [None]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_complete_file_is_only_checked() {
        let dir = crate::tests::temp_dir("download-complete");
        let path = dir.join("model.bin");
        let body = body();
        fs::write(&path, &body).unwrap();
        let server = Server::start(body.clone(), vec![]);
        assert_eq!(fetch_from(&server, &path, &body), Ok(()));
        assert_eq!(server.ranges(), Vec::<Option<String>>::new());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_file_that_fails_its_checksum_is_deleted() {
        let dir = crate::tests::temp_dir("download-checksum");
        let path = dir.join("model.bin");
        let body = body();
        let mut wrong = body.clone();
        wrong[123_456] ^= 1;
        let server = Server::start(wrong, vec![Reply::Body]);
        assert_eq!(fetch_from(&server, &path, &body), Err(Error::Checksum));
        assert!(!path.exists());
        server.ranges();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_missing_file_says_what_the_server_answered() {
        let dir = crate::tests::temp_dir("download-404");
        let path = dir.join("model.bin");
        let body = body();
        let server = Server::start(body.clone(), vec![Reply::Status(404), Reply::Status(503)]);
        let failed = |reason: &str, retry| {
            Err(Error::Failed {
                reason: reason.into(),
                retry,
            })
        };
        assert_eq!(
            fetch_from(&server, &path, &body),
            failed("the server answered 404 Not Found", false)
        );
        assert_eq!(
            fetch_from(&server, &path, &body),
            failed("the server answered 503 Service Unavailable", true)
        );
        server.ranges();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_unreachable_server_can_be_tried_again() {
        let dir = crate::tests::temp_dir("download-unreachable");
        let path = dir.join("model.bin");
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let agent = agent(Duration::from_secs(1));
        let url = format!("http://127.0.0.1:{port}/model.bin");
        let result = fetch(&agent, &url, &path, 10, &"0".repeat(64));
        assert!(
            matches!(result, Err(Error::Failed { retry: true, .. })),
            "{result:?}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_stalled_connection_times_out_and_keeps_what_arrived() {
        let dir = crate::tests::temp_dir("download-stall");
        let path = dir.join("model.bin");
        let body = body();
        let server = Server::start(body.clone(), vec![Reply::StallAfter(50_000)]);
        let start = Instant::now();
        let stalled = Error::Failed {
            reason: "timed out".into(),
            retry: true,
        };
        assert_eq!(fetch_from(&server, &path, &body), Err(stalled));
        assert!(start.elapsed() < Duration::from_secs(3));
        assert_eq!(fs::read(&path).unwrap(), &body[..50_000]);
        server.ranges();
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_untrusted_certificate_is_not_worth_trying_again() {
        let e = io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid peer certificate: UnknownIssuer",
        );
        assert_eq!(
            failed(ureq::Error::Io(e)),
            Error::Failed {
                reason: "invalid peer certificate: UnknownIssuer".into(),
                retry: false
            }
        );
    }

    #[test]
    fn reads_the_start_of_a_content_range() {
        let response = |range: &str| {
            ureq::http::Response::builder()
                .header("Content-Range", range)
                .body(ureq::Body::builder().data(""))
                .unwrap()
        };
        assert_eq!(range_start(&response("bytes 1000-1999/2000")), Some(1000));
        assert_eq!(range_start(&response("bytes */2000")), None);
        assert_eq!(range_start(&response("items 0-1/2")), None);
    }
}

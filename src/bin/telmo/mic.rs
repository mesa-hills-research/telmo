//! The microphone, through Capta. The system mixes the device down to one channel and
//! resamples it to 48 kHz. On macOS and Windows a device the system can't resample records at
//! its own rate.

use std::sync::Arc;

use crate::capture::Capture;
use crate::record::pick_device;

/// The rate to record at, the usual rate of a microphone, so the system seldom resamples.
const SAMPLE_RATE: u32 = 48_000;

/// An open input device. It listens from `open` on, and `capture` keeps what it hears between
/// a take's start and stop.
pub struct Mic {
    pub name: String,
    pub sample_rate: u32,
    pub capture: Arc<Capture>,
    _recording: capta::Capture,
}

fn input_devices() -> Result<Vec<capta::Device>, String> {
    capta::devices().map_err(|e| format!("Couldn't list the input devices ({e})."))
}

/// The names of the input devices, and whether each is the system's default.
pub fn devices() -> Result<Vec<(String, bool)>, String> {
    Ok(input_devices()?
        .into_iter()
        .map(|d| (d.name, d.is_default))
        .collect())
}

/// Opens the system's default input, or the one `device` picks by number or by part of its name.
pub fn open(device: Option<&str>) -> Result<Mic, String> {
    let device = match device {
        None => None,
        Some(pick) => {
            let mut all = input_devices()?;
            let names: Vec<String> = all.iter().map(|d| d.name.clone()).collect();
            Some(all.swap_remove(pick_device(&names, pick)?).id)
        }
    };
    let options = capta::Options {
        device,
        sample_rate: SAMPLE_RATE,
        ..capta::Options::default()
    };
    let capture = Arc::new(Capture::new());
    let (on_audio, on_error) = (Arc::clone(&capture), Arc::clone(&capture));
    let recording = capta::open(
        &options,
        move |samples, _| on_audio.push(samples.iter().copied()),
        move |error| on_error.fail(error.to_string()),
    )
    .map_err(|e| format!("Couldn't open the microphone ({e})."))?;
    Ok(Mic {
        name: recording.device().name.clone(),
        sample_rate: recording.sample_rate(),
        capture,
        _recording: recording,
    })
}

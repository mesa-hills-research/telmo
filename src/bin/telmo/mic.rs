//! The microphone, through cpal. It records at the device's own rate and sample format, mixed
//! down to one channel.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::record::{downmix, pick_device};

#[derive(Default)]
struct Shared {
    keeping: bool,
    samples: Vec<f32>,
    failure: Option<String>,
}

/// An open input device. It listens from `open` on and keeps samples between `start` and `stop`.
pub struct Mic {
    pub name: String,
    pub sample_rate: u32,
    shared: Arc<Mutex<Shared>>,
    /// The largest sample since `level` last asked, as the bits of a non-negative `f32`.
    peak: Arc<AtomicU32>,
    _stream: cpal::Stream,
}

/// The input devices that have a name.
fn input_devices(host: &cpal::Host) -> Result<Vec<(String, cpal::Device)>, String> {
    let devices = host
        .input_devices()
        .map_err(|e| format!("Couldn't list the input devices ({e})."))?;
    Ok(devices
        .filter_map(|d| d.name().ok().map(|name| (name, d)))
        .collect())
}

/// The names of the input devices, and whether each is the system's default.
pub fn devices() -> Result<Vec<(String, bool)>, String> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    Ok(input_devices(&host)?
        .into_iter()
        .map(|(name, _)| {
            let is_default = default.as_deref() == Some(name.as_str());
            (name, is_default)
        })
        .collect())
}

/// Opens the system's default input, or the one `device` picks by number or by part of its name.
pub fn open(device: Option<&str>) -> Result<Mic, String> {
    let host = cpal::default_host();
    let (name, device) = match device {
        None => {
            let device = host.default_input_device().ok_or("No microphone found.")?;
            (device.name().unwrap_or_else(|_| "unnamed".into()), device)
        }
        Some(pick) => {
            let mut all = input_devices(&host)?;
            let names: Vec<String> = all.iter().map(|(name, _)| name.clone()).collect();
            all.swap_remove(pick_device(&names, pick)?)
        }
    };
    let config = device
        .default_input_config()
        .map_err(|e| format!("Couldn't open {name} ({e})."))?;
    let format = config.sample_format();
    let config: cpal::StreamConfig = config.into();
    let shared = Arc::new(Mutex::new(Shared::default()));
    let peak = Arc::new(AtomicU32::new(0));
    let stream = match format {
        cpal::SampleFormat::F32 => build::<f32>(&device, &config, &shared, &peak),
        cpal::SampleFormat::F64 => build::<f64>(&device, &config, &shared, &peak),
        cpal::SampleFormat::I8 => build::<i8>(&device, &config, &shared, &peak),
        cpal::SampleFormat::I16 => build::<i16>(&device, &config, &shared, &peak),
        cpal::SampleFormat::I32 => build::<i32>(&device, &config, &shared, &peak),
        cpal::SampleFormat::U8 => build::<u8>(&device, &config, &shared, &peak),
        cpal::SampleFormat::U16 => build::<u16>(&device, &config, &shared, &peak),
        cpal::SampleFormat::U32 => build::<u32>(&device, &config, &shared, &peak),
        other => Err(format!(
            "{name} uses the sample format {other}, which isn't supported."
        )),
    }?;
    stream
        .play()
        .map_err(|e| format!("Couldn't start {name} ({e})."))?;
    Ok(Mic {
        name,
        sample_rate: config.sample_rate.0,
        shared,
        peak,
        _stream: stream,
    })
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: &Arc<Mutex<Shared>>,
    peak: &Arc<AtomicU32>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels.max(1));
    let (on_data, on_error) = (Arc::clone(shared), Arc::clone(shared));
    let peak = Arc::clone(peak);
    let mut mono = Vec::new();
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                mono.clear();
                downmix(
                    data.iter().map(|s| s.to_sample::<f32>()),
                    channels,
                    &mut mono,
                );
                let loudest = mono.iter().fold(0.0_f32, |m, s| m.max(s.abs()));
                peak.fetch_max(loudest.to_bits(), Ordering::Relaxed);
                let mut shared = on_data.lock().unwrap_or_else(|e| e.into_inner());
                if shared.keeping {
                    shared.samples.extend_from_slice(&mono);
                }
            },
            move |error| {
                let mut shared = on_error.lock().unwrap_or_else(|e| e.into_inner());
                shared.failure.get_or_insert(error.to_string());
            },
            None,
        )
        .map_err(|e| format!("Couldn't open the microphone ({e})."))
}

impl Mic {
    /// Starts keeping samples, dropping any kept before.
    pub fn start(&self) {
        let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.samples.clear();
        shared.keeping = true;
    }

    /// Stops keeping samples and returns them.
    pub fn stop(&self) -> Vec<f32> {
        let mut shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.keeping = false;
        std::mem::take(&mut shared.samples)
    }

    /// The largest sample, from 0 to 1, since the last call.
    pub fn level(&self) -> f32 {
        f32::from_bits(self.peak.swap(0, Ordering::Relaxed))
    }

    /// Why the device stopped, if it did.
    pub fn failure(&self) -> Option<String> {
        let shared = self.shared.lock().unwrap_or_else(|e| e.into_inner());
        shared.failure.clone()
    }
}

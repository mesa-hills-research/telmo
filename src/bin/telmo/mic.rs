//! The microphone, through cpal. It records at the device's own rate and sample format, mixed
//! down to one channel.

use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::capture::Capture;
use crate::record::pick_device;

/// An open input device. It listens from `open` on, and `capture` keeps what it hears between
/// a take's start and stop.
pub struct Mic {
    pub name: String,
    pub sample_rate: u32,
    pub capture: Arc<Capture>,
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
    let capture = Arc::new(Capture::new());
    let stream = match format {
        cpal::SampleFormat::F32 => build::<f32>(&device, &config, &capture),
        cpal::SampleFormat::F64 => build::<f64>(&device, &config, &capture),
        cpal::SampleFormat::I8 => build::<i8>(&device, &config, &capture),
        cpal::SampleFormat::I16 => build::<i16>(&device, &config, &capture),
        cpal::SampleFormat::I32 => build::<i32>(&device, &config, &capture),
        cpal::SampleFormat::U8 => build::<u8>(&device, &config, &capture),
        cpal::SampleFormat::U16 => build::<u16>(&device, &config, &capture),
        cpal::SampleFormat::U32 => build::<u32>(&device, &config, &capture),
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
        capture,
        _stream: stream,
    })
}

fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    capture: &Arc<Capture>,
) -> Result<cpal::Stream, String>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    let channels = usize::from(config.channels.max(1));
    let (on_data, on_error) = (Arc::clone(capture), Arc::clone(capture));
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                on_data.push(data.iter().map(|s| s.to_sample::<f32>()), channels);
            },
            move |error| on_error.fail(error.to_string()),
            None,
        )
        .map_err(|e| format!("Couldn't open the microphone ({e})."))
}

extern crate find_folder;
extern crate hound;
extern crate sample;
extern crate rand;

use sample::{signal, Signal};

use rand::{Rng, SeedableRng, StdRng};


pub const NUM_CHANNELS: usize = 2;
pub type Frame = [i16; NUM_CHANNELS];

const SAMPLE_RATE: f64 = 44_100.0;

#[macro_use]
extern crate clap;
use clap::{App, Arg};

fn main() {
    let matches = App::new("markov_jukebox")
        .version(crate_version!())
        .arg(
            Arg::with_name("filenames")
                .multiple(true)
                .help("the path(s) of the file(s) you wish to blend together.")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("keep")
                .short("k")
                .help("keep used samples available for repeat uses."),
        )
        .arg(
            Arg::with_name("seed")
                .short("s")
                .takes_value(true)
                .help("the seed for the pseudo-random generator"),
        )
        .arg(Arg::with_name("debug").short("d").help(
            "just write files directly to output. overrides all other flags",
        ))
        .get_matches();

    //there used to be a `play` option. That's why this is a struct instead of a bool
    let settings = Settings {
        blend: !matches.is_present("debug"),
        remove: !matches.is_present("keep"),
    };

    let seed: Vec<usize> = if let Some(passed_seed) = matches.value_of("seed") {
        passed_seed.as_bytes().iter().map(|&b| b as usize).collect()
    } else {
        if cfg!(debug_assertions) {
            vec![42]
        } else {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|dur| dur.as_secs())
                .unwrap_or(42);
            vec![timestamp as usize]
        }
    };
    let mut rng: StdRng = SeedableRng::from_seed(seed.as_slice());

    if let Some(filenames) = matches.values_of("filenames") {
        run(filenames.collect(), settings, &mut rng).unwrap();
    } else {
        println!("No filenames recieved");
    }

}

struct Settings {
    pub remove: bool,
    pub blend: bool,
}

fn run<R: Rng>(filenames: Vec<&str>, settings: Settings, rng: &mut R) -> Result<(), ()> {
    let frames: Vec<Frame> = {
        let mut frames = Vec::new();

        for filename in filenames.iter() {
            let current_frames: Vec<Frame> = read_frames(filename);

            frames.extend(&current_frames);
        }

        frames
    };

    if settings.blend {
        let blended_frames = blend_frames(&frames, rng, settings.remove);
        write_frames(&blended_frames, None);
    } else {
        write_frames(&frames, None);
    }

    Ok(())
}

// Given the file name, produces a Vec of `Frame`s which may be played back.
fn read_frames(file_name: &str) -> Vec<Frame> {
    println!("Loading {}", file_name);

    let mut reader = hound::WavReader::open(file_name).unwrap();
    let spec = reader.spec();

    assert!(spec.channels > 0, "{} says it has 0 channels?!", file_name);
    let duration = reader.duration();
    let new_duration = (duration as f64 * (SAMPLE_RATE as f64 / spec.sample_rate as f64)) as usize;
    let samples: Box<Iterator<Item = i16>> = if spec.bits_per_sample <= 16 {
        Box::new(reader.samples().map(|s| s.unwrap()))
    } else {
        Box::new(reader.samples::<f32>().map(|s| {
            let f: f32 = s.unwrap();

            (f * 32768.0) as i16
        }))
    };

    let adjusted_samples: Vec<_> = if spec.channels == 2 {
        samples.collect()
    } else if spec.channels <= 1 {
        samples.flat_map(|s| vec![s, s]).collect()
    } else {
        samples
            .enumerate()
            .filter(|&(i, _)| i % (spec.channels as usize) < 2)
            .map(|(_, s)| s)
            .collect()
    };

    let signal = signal::from_interleaved_samples::<_, Frame>(adjusted_samples.iter().cloned());

    signal
        .from_hz_to_hz(spec.sample_rate as f64, SAMPLE_RATE as f64)
        .take(new_duration)
        .collect()
}

const SILENCE: Frame = [0; NUM_CHANNELS];

fn blend_frames<R: Rng>(frames: &Vec<Frame>, rng: &mut R, remove: bool) -> Vec<Frame> {
    let len = frames.len();

    if len == 0 {
        return Vec::new();
    }

    println!("get_next_frames");

    let mut next_frames = get_next_frames(frames);

    println!("done get_next_frames");

    let mut result = Vec::with_capacity(len);

    let default = vec![SILENCE];

    let mut previous = (frames[0], frames[1]);
    for i in 0..2 {
        result.push(frames[i]);
    }

    let mut keys: Vec<(Frame, Frame)> = next_frames.keys().map(|&k| k).collect();

    println!("sorting {}", keys.len());
    keys.sort();
    keys.reverse();

    println!("shuffling");
    rng.shuffle(&mut keys);

    let mut count = 0;
    let mut missed_count = 0;

    let enough = if remove { (len * 3) / 4 } else { len };

    while count < enough {
        let choices = if remove {
            next_frames
                .remove(&previous)
                .and_then(|c| if c.len() > 0 { Some(c) } else { None })
                .unwrap_or_else(|| {
                    if cfg!(debug_assertions) {
                        println!("default at {}", count);
                    }
                    default.clone()
                })
        } else {
            (*next_frames
                .get(&previous)
                .and_then(|c| if c.len() > 0 { Some(c) } else { None })
                .unwrap_or_else(|| {
                    if cfg!(debug_assertions) {
                        println!("default at {}", count);
                    }
                    &default
                })).clone()
        };

        let next = *rng.choose(&choices).unwrap();

        if is_audible(&next) || is_audible(&previous.0) || is_audible(&previous.1) {
            result.push(next);
            missed_count = 0;
        } else {
            missed_count += 1;
        }

        if missed_count > 16 {
            previous = *rng.choose(&keys).unwrap();
            println!("rng.choose => {:?}", previous);
        } else {
            previous = (previous.1, next);
        }


        count += 1;
    }

    result
}

use std::collections::HashMap;

type NextFrames = HashMap<(Frame, Frame), Vec<Frame>>;

fn get_next_frames(frames: &Vec<Frame>) -> NextFrames {
    let mut result = HashMap::new();

    for window in frames.windows(3) {
        result
            .entry((window[0], window[1]))
            .or_insert(Vec::new())
            .push(window[2]);
    }

    result
}

fn is_audible(frame: &Frame) -> bool {
    magnitude(frame) > 128
}

use std::cmp::max;

fn magnitude(frame: &Frame) -> i16 {

    frame
        .iter()
        .fold(0, |acc, channel_value| max(acc, channel_value.abs()))
}

fn write_frames(frames: &Vec<Frame>, optional_name: Option<&str>) {
    if let Some(name) = optional_name {
        write_frames_with_name(frames, name)
    } else {
        let name = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string();

        write_frames_with_name(frames, &name)
    }
}

fn write_frames_with_name(frames: &Vec<Frame>, name: &str) {
    let mut path = std::path::PathBuf::new();
    path.push("output");
    path.push(name);
    path.set_extension("wav");

    println!("Writing to {:?}", path.to_str().unwrap());

    let spec = hound::WavSpec {
        channels: NUM_CHANNELS as _,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for frame in frames.iter() {
        for channel in 0..NUM_CHANNELS {
            writer.write_sample(frame[channel]).unwrap();
        }
    }

    writer.finalize().unwrap();
}

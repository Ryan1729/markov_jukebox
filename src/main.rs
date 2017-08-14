extern crate find_folder;
extern crate hound;
extern crate portaudio as pa;
extern crate sample;
extern crate rand;

use sample::{signal, Signal, ToFrameSliceMut};

use rand::{Rng, StdRng, SeedableRng};

mod wav {
    pub const NUM_CHANNELS: usize = 2;
    pub const PATH: &'static str = "narrow.wav";
    pub type Frame = [i16; NUM_CHANNELS];
}

const FRAMES_PER_BUFFER: u32 = 64;
const SAMPLE_RATE: f64 = 44_100.0;


extern crate clap;
use clap::{Arg, App};

fn main() {
    let matches = App::new("markov_jukebox")
        .arg(Arg::with_name("filenames").takes_value(true).required(true))
        .arg(Arg::with_name("play").short("p").help(
            "play the files before processing them",
        ))
        .get_matches();

    let play = matches.is_present("play");

    let seed: &[_] = &[42];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    if let Some(filenames) = matches.values_of("filenames") {
        run(filenames.collect(), play, &mut rng).unwrap();
    } else {
        println!("No filenames recieved");
    }

}

fn run<R: Rng>(filenames: Vec<&str>, play: bool, rng: &mut R) -> Result<(), pa::Error> {
    if play {
        // Initialise PortAudio.
        let pa = try!(pa::PortAudio::new());
        let settings = try!(pa.default_output_stream_settings::<i16>(
            wav::NUM_CHANNELS as i32,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
        ));

        for filename in filenames.iter() {
            // Get the frames to play back.
            let frames: Vec<wav::Frame> = read_frames(filename);
            let mut signal = frames.clone().into_iter();

            // Define the callback which provides PortAudio the audio.
            let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
                let buffer: &mut [wav::Frame] = buffer.to_frame_slice_mut().unwrap();
                for out_frame in buffer {
                    match signal.next() {
                        Some(frame) => *out_frame = frame,
                        None => return pa::Complete,
                    }
                }
                pa::Continue
            };

            let mut stream = try!(pa.open_non_blocking_stream(settings, callback));
            try!(stream.start());

            while let Ok(true) = stream.is_active() {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            try!(stream.stop());
            try!(stream.close());

            let blended_frames = blend_frames(&frames, rng);
            write_frames(&blended_frames, None);
        }
    } else {
        for filename in filenames.iter() {
            let frames: Vec<wav::Frame> = read_frames(filename);
            let blended_frames = blend_frames(&frames, rng);
            write_frames(&blended_frames, None);
        }
    }


    Ok(())
}

// Given the file name, produces a Vec of `Frame`s which may be played back.
fn read_frames(file_name: &str) -> Vec<wav::Frame> {
    println!("Loading {}", file_name);

    let mut reader = hound::WavReader::open(file_name).unwrap();
    let spec = reader.spec();
    let duration = reader.duration();
    let new_duration = (duration as f64 * (SAMPLE_RATE as f64 / spec.sample_rate as f64)) as usize;
    let samples = reader.samples().map(|s| s.unwrap());
    let signal = signal::from_interleaved_samples::<_, wav::Frame>(samples);
    signal
        .from_hz_to_hz(spec.sample_rate as f64, SAMPLE_RATE as f64)
        .take(new_duration)
        .collect()
}

const SILENCE: wav::Frame = [0; wav::NUM_CHANNELS];

fn blend_frames<R: Rng>(frames: &Vec<wav::Frame>, rng: &mut R) -> Vec<wav::Frame> {
    let len = frames.len();

    if len == 0 {
        return Vec::new();
    }

    let next_frames = get_next_frames(frames);

    let mut result = Vec::with_capacity(len);

    let default = vec![SILENCE];

    let (first, second) = (frames[0], frames[1]);
    let mut previous = (key_round_frame(first), key_round_frame(second));
    result.push(first);
    result.push(second);

    let mut count = 0;
    while count < len {
        let choices = next_frames
            .get(&previous)
            .and_then(|c| if c.len() > 0 { Some(c) } else { None })
            .unwrap_or(&default);
        let next = rng.choose(&choices).unwrap();

        result.push(*next);

        previous = (key_round_frame(previous.1), key_round_frame(*next));

        count += 1;
    }

    result
}

use std::collections::HashMap;

fn get_next_frames(frames: &Vec<wav::Frame>) -> HashMap<(wav::Frame, wav::Frame), Vec<wav::Frame>> {
    let mut result = HashMap::new();

    for window in frames.windows(3) {
        result
            .entry((key_round_frame(window[0]), key_round_frame(window[1])))
            .or_insert(Vec::new())
            .push(window[2]);
    }

    result
}

fn key_round_frame(frame: wav::Frame) -> wav::Frame {
    [key_round(frame[0]), key_round(frame[1])]
}

fn key_round(x: i16) -> i16 {
    // if x & 0b10 == 0 {
    //     x & !0b11
    // } else {
    //     (x | 0b11).saturating_add(1)
    // }
    if x & 1 == 0 { x } else { (x).saturating_add(1) }

}

fn write_frames(frames: &Vec<wav::Frame>, optional_name: Option<&str>) {
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

fn write_frames_with_name(frames: &Vec<wav::Frame>, name: &str) {
    let mut path = std::path::PathBuf::new();
    path.push("output");
    path.push(name);
    path.set_extension("wav");

    println!("Writing to {:?}", path.to_str().unwrap());

    let spec = hound::WavSpec {
        channels: wav::NUM_CHANNELS as _,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    for frame in frames.iter() {
        for channel in 0..wav::NUM_CHANNELS {
            writer.write_sample(frame[channel]).unwrap();
        }
    }

    writer.finalize().unwrap();
}

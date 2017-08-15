extern crate find_folder;
extern crate hound;
extern crate portaudio as pa;
extern crate sample;
extern crate rand;

use sample::{signal, Signal, ToFrameSliceMut};

use rand::{Rng, StdRng, SeedableRng};


pub const NUM_CHANNELS: usize = 2;
pub type Frame = [i16; NUM_CHANNELS];

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
            NUM_CHANNELS as i32,
            SAMPLE_RATE,
            FRAMES_PER_BUFFER,
        ));

        for filename in filenames.iter() {
            // Get the frames to play back.
            let frames: Vec<Frame> = read_frames(filename);
            let mut signal = frames.clone().into_iter();

            // Define the callback which provides PortAudio the audio.
            let callback = move |pa::OutputStreamCallbackArgs { buffer, .. }| {
                let buffer: &mut [Frame] = buffer.to_frame_slice_mut().unwrap();
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
            let frames: Vec<Frame> = read_frames(filename);
            let blended_frames = blend_frames(&frames, rng);
            write_frames(&blended_frames, None);
        }
    }


    Ok(())
}

// Given the file name, produces a Vec of `Frame`s which may be played back.
fn read_frames(file_name: &str) -> Vec<Frame> {
    println!("Loading {}", file_name);

    let mut reader = hound::WavReader::open(file_name).unwrap();
    let spec = reader.spec();
    let duration = reader.duration();
    let new_duration = (duration as f64 * (SAMPLE_RATE as f64 / spec.sample_rate as f64)) as usize;
    let samples = reader.samples().map(|s| s.unwrap());
    let signal = signal::from_interleaved_samples::<_, Frame>(samples);
    signal
        .from_hz_to_hz(spec.sample_rate as f64, SAMPLE_RATE as f64)
        .take(new_duration)
        .collect()
}

const SILENCE: Frame = [0; NUM_CHANNELS];

fn blend_frames<R: Rng>(frames: &Vec<Frame>, rng: &mut R) -> Vec<Frame> {
    let len = frames.len();

    if len == 0 {
        return Vec::new();
    }

    println!("get_next_frames");

    let next_frames = get_next_frames(frames);

    println!("done get_next_frames");

    let mut result = Vec::with_capacity(len);

    let default = vec![SILENCE];

    let mut previous = (frames[0], frames[1]);
    for i in 0..2 {
        result.push(frames[i]);
    }

    rng.gen_range(0, 12);

    let mut count = 0;
    println!("{} < {}   {} ", count, len, result.len());
    while count < len {
        let choices = get_choices(&next_frames, previous);

        let next = *rng.choose(&choices).unwrap();

        result.push(next);

        previous = (previous.1, next);

        count += 1;
    }
    println!("{}", result.len());
    result
}

const MINIMUM_CHOICES: usize = 5;

use std::collections::HashSet;

fn get_choices(next_frames: &NextFrames, previous: (Frame, Frame)) -> Vec<Frame> {
    let mut seen = HashSet::new();

    //order is not important beyond BFS, so FIFO will do
    let mut queue = Vec::new();

    let mut result = Vec::new();

    seen.insert(previous);
    queue.push(previous);

    while let Some(current) = queue.pop() {
        if let Some(choices) = next_frames.get(&current) {
            result.extend(choices);
        }

        if result.len() >= MINIMUM_CHOICES {
            return result;
        }

        let offsets = [(0, 1), (-1, 0), (0, -1), (1, 0)];

        for &(offset_0, offset_1) in offsets.iter() {
            let new_key = (
                saturating_add(previous.0, offset_0),
                saturating_add(previous.1, offset_1),
            );

            queue.push(new_key)
        }


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

fn saturating_add(frame: Frame, x: i16) -> Frame {
    [frame[0].saturating_add(x), frame[1].saturating_add(x)]
}
fn saturating_sub(frame: Frame, x: i16) -> Frame {
    [frame[0].saturating_sub(x), frame[1].saturating_sub(x)]
}


fn key_round_frame(frame: Frame) -> Frame {
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

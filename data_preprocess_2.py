import os
import music21 as m21
import numpy as np
import csv
from pathlib import Path

def extract_absolute_pitch_and_rhythm(midi_file):
    """Extract pitch and symbolic rhythm sequences using music21."""
    try:
        # Load MIDI file
        score = m21.converter.parse(midi_file)
        
        absolute_pitch_sequence = []
        absolute_rhythm_sequence = []
        
        # Process each part in the score
        for part in score.parts:
            for note in part.flatten().notesAndRests:
                if isinstance(note, m21.note.Note):  # Skip rests
                    # Extract pitch
                    absolute_pitch_sequence.append(note.pitch.midi)
                    
                    # Convert duration to symbolic label
                    duration_type = note.duration.type
                    if duration_type in ['whole', 'half', 'quarter', 'eighth', '16th', '32nd', 'dotted half', 'dotted quarter', 'dotted eighth']:
                        absolute_rhythm_sequence.append(duration_type)
                    else:
                        # Handle complex durations by converting to quarter note ratio
                        quarter_length = note.duration.quarterLength
                        if quarter_length >= 4.0:
                            absolute_rhythm_sequence.append('whole')
                        elif quarter_length >= 2.0:
                            absolute_rhythm_sequence.append('half')
                        elif quarter_length >= 1.0:
                            absolute_rhythm_sequence.append('quarter')
                        elif quarter_length >= 0.5:
                            absolute_rhythm_sequence.append('eighth')
                        elif quarter_length >= 0.25:
                            absolute_rhythm_sequence.append('16th')
                        elif quarter_length >= 0.125:
                            absolute_rhythm_sequence.append('32nd')
                        elif quarter_length >= 3.0:
                            absolute_rhythm_sequence.append('dotted half')
                        elif quarter_length >= 1.5:
                            absolute_rhythm_sequence.append('dotted quarter')
                        elif quarter_length >= 0.75:
                            absolute_rhythm_sequence.append('dotted eighth')

        if len(absolute_pitch_sequence) == 0 or len(absolute_rhythm_sequence) == 0:
            raise ValueError("No valid pitch or rhythm data found.")

        return np.array(absolute_pitch_sequence), np.array(absolute_rhythm_sequence)
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return None, None

def compute_relative_pitch(pitch_sequence):
    """Calculate intervals between consecutive pitches."""
    return np.diff(pitch_sequence)

def compute_relative_rhythm(rhythm_sequence):
    """
    Convert symbolic rhythm sequence to relative durations.
    Maps each pair of consecutive durations to their relative length ratio.
    """
    # Duration mapping in terms of quarter notes
    duration_map = {
        'whole': 4.0,
        'half': 2.0,
        'quarter': 1.0,
        'eighth': 0.5,
        '16th': 0.25,
        '32nd': 0.125,
        'dotted half': 3.0,
        'dotted quarter': 1.5,
        'dotted eighth': 0.75
    }
    
    # Convert symbolic durations to numeric values
    numeric_durations = np.array([duration_map[r] for r in rhythm_sequence])
    
    # Calculate ratio between consecutive durations
    relative_rhythm = np.array([numeric_durations[i+1] / numeric_durations[i] 
                              for i in range(len(numeric_durations) - 1)])
    return np.around(relative_rhythm, decimals=3)

def sliding_window(sequence, window_size=5, step_size=1):
    """Split sequence into overlapping windows."""
    windows = [
        sequence[i : i + window_size]
        for i in range(0, len(sequence) - window_size + 1, step_size)
    ]
    return np.array(windows)

def get_file_info(file_path, base_folder):
    """Extract file information."""
    rel_path = os.path.relpath(os.path.dirname(file_path), base_folder)
    folders = rel_path.split(os.sep)
    case_info = next((folder for folder in folders if "case" in folder.lower()), "Unknown Case")
    ruling = "Plagiarism" if "Plagiarism" in base_folder and "No_Plagiarism" not in base_folder else "No_Plagiarism"
    
    return {
        "ruling": ruling,
        "case": case_info,
        "subfolder_path": rel_path,
        "filename": os.path.basename(file_path)
    }

def store_library(library_path, data):
    """Save library with symbolic rhythm values and separate sequence counts."""
    with open(library_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File Name", "Ruling", "Case",
            "Absolute Pitch", "Absolute Rhythm", 
            "Relative Pitch", "Relative Rhythm",
            "Absolute Pitch Sequences", "Absolute Rhythm Sequences",
            "Relative Pitch Sequences", "Relative Rhythm Sequences"
        ])
        for file_name, info in data.items():
            # Count sequences for each feature
            abs_pitch_count = len(info["absolute_pitch"])
            abs_rhythm_count = len(info["absolute_rhythm"])
            rel_pitch_count = len(info["relative_pitch"])
            rel_rhythm_count = len(info["relative_rhythm"])
            
            writer.writerow([
                file_name,
                info["file_info"]["ruling"],
                info["file_info"]["case"],
                info["absolute_pitch"].tolist(),
                info["absolute_rhythm"].tolist(),
                info["relative_pitch"].tolist(),
                info["relative_rhythm"].tolist(),
                abs_pitch_count,
                abs_rhythm_count,
                rel_pitch_count,
                rel_rhythm_count
            ])

def store_library_unsegmented(library_path, data):
    """Store unsegmented library with one sequence per MIDI file."""
    with open(library_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "File Name", "Ruling", "Case",
            "Absolute Pitch", "Absolute Rhythm", 
            "Relative Pitch", "Relative Rhythm"
        ])
        
        for file_name, info in data.items():
            # Use original unsegmented sequences
            writer.writerow([
                file_name,
                info["file_info"]["ruling"],
                info["file_info"]["case"],
                info["original_pitch"].tolist(),
                info["original_rhythm"].tolist(),
                info["original_rel_pitch"].tolist(),
                info["original_rel_rhythm"].tolist()
            ])

def process_file(midi_path, abs_output_dir, rel_output_dir, file_name, library, file_info, window_size, step_size):
    """Process individual MIDI file with symbolic rhythm values."""
    try:
        # Extract complete sequences first
        original_pitch, original_rhythm = extract_absolute_pitch_and_rhythm(midi_path)
        
        if original_pitch is None or original_rhythm is None:
            return
            
        # Calculate relative sequences for the complete melody
        original_rel_pitch = compute_relative_pitch(original_pitch)
        original_rel_rhythm = compute_relative_rhythm(original_rhythm)
        
        # Now apply sliding window for segmented analysis
        pitch_segments = sliding_window(original_pitch, window_size=window_size, step_size=step_size)
        rhythm_segments = sliding_window(original_rhythm, window_size=window_size, step_size=step_size)
        
        if len(pitch_segments) == 0 or len(rhythm_segments) == 0:
            return
        
        # Save segmented sequences
        pitch_file = os.path.join(abs_output_dir, f"{file_name}_pitch.npy")
        rhythm_file = os.path.join(abs_output_dir, f"{file_name}_rhythm.npy")
        np.save(pitch_file, pitch_segments)
        np.save(rhythm_file, rhythm_segments)
        
        os.makedirs(rel_output_dir, exist_ok=True)
        
        # Process relative sequences for segments
        relative_pitch = []
        relative_rhythm = []
        
        for pitch_seg in pitch_segments:
            rel_pitch = compute_relative_pitch(pitch_seg)
            if len(rel_pitch) > 0:
                relative_pitch.append(rel_pitch)
        
        for rhythm_seg in rhythm_segments:
            rel_rhythm = compute_relative_rhythm(rhythm_seg)
            if len(rel_rhythm) > 0:
                relative_rhythm.append(rel_rhythm)
        
        relative_pitch = np.array(relative_pitch)
        relative_rhythm = np.array(relative_rhythm)
        
        if len(relative_pitch) > 0 and len(relative_rhythm) > 0:
            rel_pitch_file = os.path.join(rel_output_dir, f"{file_name}_relative_pitch.npy")
            rel_rhythm_file = os.path.join(rel_output_dir, f"{file_name}_relative_rhythm.npy")
            np.save(rel_pitch_file, relative_pitch)
            np.save(rel_rhythm_file, relative_rhythm)
            
            library[file_name] = {
                "file_info": file_info,
                "absolute_pitch": pitch_segments,
                "absolute_rhythm": rhythm_segments,
                "relative_pitch": relative_pitch,
                "relative_rhythm": relative_rhythm,
                "original_pitch": original_pitch,
                "original_rhythm": original_rhythm,
                "original_rel_pitch": original_rel_pitch,
                "original_rel_rhythm": original_rel_rhythm
            }
        
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")

def process_folder(input_folder, abs_output_folder, rel_output_folder, library_path):
    """Process entire folder structure with music21-based extraction."""
    library = {}
    
    # Get user parameters for sliding window
    window_size, step_size = get_sequence_parameters()
    print(f"\nWith window size: {window_size}, step size: {step_size}, we obtain the n-gram length of window size -1 for the relative pitch and rhythm sequences.")
    
    for ruling in ["Plagiarism", "No_Plagiarism"]:
        ruling_input = os.path.join(input_folder, ruling)
        if not os.path.exists(ruling_input):
            print(f"Error: Ruling folder not found - {ruling_input}")
            continue
            
        for root, _, files in os.walk(ruling_input):
            midi_files = [f for f in files if f.endswith('.mid')]
            if not midi_files:
                continue
                
            for file in midi_files:
                midi_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, ruling_input)
                abs_output_dir = os.path.join(abs_output_folder, ruling, rel_path)
                rel_output_dir = os.path.join(rel_output_folder, ruling, rel_path)
                
                os.makedirs(abs_output_dir, exist_ok=True)
                os.makedirs(rel_output_dir, exist_ok=True)
                
                file_info = get_file_info(midi_path, os.path.join(input_folder, ruling))
                process_file(midi_path, abs_output_dir, rel_output_dir, file[:-4], 
                           library, file_info, window_size, step_size)
    
    store_library(library_path, library)
    unsegmented_path = os.path.join(os.path.dirname(library_path), "unsegmented_melody_library_symbolic.csv")
    store_library_unsegmented(unsegmented_path, library)
    
    print(f"\nProcessing completed:")
    print(f"- Window size: {window_size}")
    print(f"- Step size: {step_size}")
    print(f"- Segmented library: {library_path}")
    print(f"- Unsegmented library: {unsegmented_path}")

def get_sequence_parameters():
    """Get user input for sliding window parameters."""
    while True:
        print("\nSelect window size for data preprocessing:")
        print("Window size = 3")
        print("Window size = 4")
        print("Window size = 5")
        print("Window size = 6")
        print("Window size = 7")
        
        try:
            choice = int(input("\nEnter your choice (3-7): "))
            if 1 <= choice <= 7:
                window_size = choice
                break
            print("Invalid choice! Please select 3-7.")
        except ValueError:
            print("Invalid input! Please enter a number between 3-7.")
    
    while True:
        print("\nSelect step size for sliding window:")
        print("1. Step size = 1 (Maximum overlap)")
        print("2. Step size = 2")
        print("3. Step size = 3")
        print("4. Step size = 4")
        print("5. Step size = 5 (Minimum overlap)")
        
        try:
            choice = int(input("\nEnter your choice (1-5): "))
            if 1 <= choice <= 5:
                step_size = choice
                break
            print("Invalid choice! Please select 1-5.")
        except ValueError:
            print("Invalid input! Please enter a number between 1-5.")
    
    return window_size, step_size

def check_directories():
    """Check and verify required directory structure."""
    dataset_folder = os.path.join(os.path.dirname(__file__), "MCIC_Dataset")
    input_folder = os.path.join(dataset_folder, "MCIC_Raw")
    
    plagiarism_input = os.path.join(input_folder, "Plagiarism")
    no_plagiarism_input = os.path.join(input_folder, "No_Plagiarism")
    
    directories = {
        'dataset': dataset_folder,
        'input': input_folder,
        'input_plagiarism': plagiarism_input,
        'input_no_plagiarism': no_plagiarism_input
    }
    
    missing = []
    for name, path in directories.items():
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print("Error: Required input directories not found:")
        for path in missing:
            print(f"  - {path}")
        print("\nPlease ensure the following directory structure exists:")
        print("MelDet/")
        print("└── MCIC_Dataset/")
        print("    ├── MCIC_Raw/")
        print("    │   ├── Plagiarism/")
        print("    │   └── No_Plagiarism/")
        print("    └── MCIC_Preprocessed/")
        exit(1)
    
    return input_folder

if __name__ == "__main__":
    # Define root directories
    dataset_folder = os.path.join(os.path.dirname(__file__), "MCIC_Dataset")
    input_folder = os.path.join(dataset_folder, "MCIC_Raw")
    output_folder = os.path.join(dataset_folder, "MCIC_Preprocessed")
    
    # Verify directory structure
    input_folder = check_directories()
    
    # Process files with music21
    process_folder(
        input_folder,
        os.path.join(output_folder, "absolute_sequences_symbolic"),
        os.path.join(output_folder, "relative_intervals_symbolic"),
        os.path.join(output_folder, "melody_library_symbolic.csv")
    )
    
    print("\nSymbolic rhythm processing completed.")
    print(f"Preprocessed data saved in {output_folder}")
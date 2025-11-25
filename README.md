# training-free_activity_recognition
![图](https://github.com/user-attachments/assets/a2215d24-171e-429e-bd11-c8550d29e882)
## Instructions

Download the pre-trained weights for ActionCLIP(https://github.com/sallymmx/ActionCLIP).

## Generate Few-Shot Data
Run the encode_dataset.py script to generate few-shot data.

## Generate Text Features
Run the generate_text_classifier_weights.py script to generate text features.

## Test Zero-Shot Recognition
Run the run_zs_baseline.py script to test the zero-shot recognition performance.

## Test the Improved TIP-Adapter Method
Run the tip_adaper.py script to test the improved TIP-Adapter method for building activity recognition.

## Test the Improved TIP-X Method
Run the tipx.py script to test the improved TIP-X method for building activity recognition.

## Test the Proposed Method
Run the tipx-feed.py script to test the proposed method for building activity recognition.

## Acknowledgments
Our code is based on ActionCLIP, TIP-Adapter and SUS-X.

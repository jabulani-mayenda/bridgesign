# SMART SIGN Training Schedule

Use this schedule to improve recognition by collecting better data, not just more data. The goal is a balanced dataset with different lighting, distances, camera types, hands, skin tones, backgrounds, and signing speeds.

## Targets

- Static alphabet signs: 300+ samples per letter.
- High-confusion static signs: 450+ samples each for `A`, `B`, `E`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`.
- Motion letters: 250+ gesture samples each for `J` and `Z`.
- High-priority word signs: 250+ gesture samples each for `HELP`, `STOP`, `WATER`, `THANK_YOU`, `YES`, `NO`, `PLEASE`, `SORRY`.
- Negative/idle class: 250+ gesture samples for `STATIC`.

## Sample Mix Per Sign

For every sign you collect, split the samples like this:

- 25% normal room light, plain background.
- 20% dim light.
- 20% bright/window light.
- 15% different distance from camera.
- 10% slight left/right body angle.
- 10% different camera or different person, when available.

Do not collect all samples while holding the exact same pose. Pause, reset your hand, and vary the position naturally.

## Daily 45 Minute Loop

1. Run the report:
   `python tools/daily_accuracy_report.py`

2. Collect 15 minutes of static signs:
   Use the report's `data_collector.py` command first. Prioritize the lowest-count labels and signs you personally see failing in Live/Image mode.

3. Collect 15 minutes of motion signs:
   Use the report's `gesture_collector.py` command. Prioritize `J`, `Z`, `STATIC`, `HELP`, `STOP`, `WATER`, and `THANK_YOU`.

4. Retrain:
   `python model_trainer.py`
   `python gesture_trainer.py`
   `python lstm_gesture_trainer.py`

5. Audit:
   `python _system_audit.py`

6. Smoke test:
   In Live, test 5 weak letters, `J`, `Z`, `HELP`, and `STOP`.
   In Image, upload one clear front-camera selfie photo and one back-camera photo.
   In Call, test one room with one side set to `I speak - voice to signs` and the other to `I sign - camera to speech`.

## 4 Week Plan

### Week 1: Balance The Alphabet

Goal: every static letter reaches at least 300 samples.

Day 1:
`python tools/daily_accuracy_report.py`
Collect the 8 lowest-count letters.

Day 2:
Collect the next 8 lowest-count letters.

Day 3:
Collect the final 10 letters.

Day 4:
Collect confusion pairs: `A E M N S T`.

Day 5:
Collect directional/angle-sensitive signs: `P Q R K V X`.

Day 6:
Retrain, audit, then manually test every letter A-Z.

Day 7:
Rest or only collect labels that failed during manual testing.

### Week 2: Lighting And Camera Robustness

Goal: make Live and Image mode work in realistic bad conditions.

Day 1:
Dim room collection for the 8 weakest static signs.

Day 2:
Bright/window light collection for the 8 weakest static signs.

Day 3:
Back-camera phone style: farther from camera, full hand visible.

Day 4:
Front-camera selfie style: close to camera, natural hand position.

Day 5:
Busy but non-distracting background. Avoid faces or clutter directly behind the hand.

Day 6:
Retrain and test Image mode using saved photos from different lighting.

Day 7:
Only recollect signs that still fail in Image mode.

### Week 3: Motion And Word Signs

Goal: improve signs that need movement or word-level output.

Day 1:
`python gesture_collector.py --batch J Z STATIC --samples 250`

Day 2:
`python gesture_collector.py --batch HELP STOP WATER --samples 250`

Day 3:
`python gesture_collector.py --batch THANK_YOU YES NO --samples 250`

Day 4:
`python gesture_collector.py --batch PLEASE SORRY --samples 250`

Day 5:
Collect slow, normal, and fast versions of the weakest motion signs.

Day 6:
Retrain all models and test Live word/sign flow.

Day 7:
Call module test: one person signs, one person speaks. Record which signs fail.

### Week 4: Real User Testing

Goal: stop training only on your own ideal examples.

Day 1:
Collect samples from another person if possible.

Day 2:
Collect using a phone camera and laptop camera.

Day 3:
Collect seated and standing samples.

Day 4:
Collect natural-speed signing, not perfect classroom poses.

Day 5:
Run a full retrain and save the metrics.

Day 6:
Presentation simulation: Live, Image, Speech, and Call.

Day 7:
Freeze the model if accuracy is good. Only fix serious failures after this.

## Collection Commands

Static signs:
`python data_collector.py --batch A B C D E F G H --samples 300`

Weak static signs:
`python data_collector.py --batch A B E M N P Q R S T --samples 450`

Motion signs:
`python gesture_collector.py --batch J Z STATIC HELP STOP WATER THANK_YOU YES NO PLEASE SORRY --samples 250`

Retrain:
`python model_trainer.py`
`python gesture_trainer.py`
`python lstm_gesture_trainer.py`

Report:
`python tools/daily_accuracy_report.py`

Audit:
`python _system_audit.py`

## Rules For Good Data

- Keep the whole hand in frame.
- Record from wrist to fingertips.
- Avoid motion blur.
- Do not collect 300 identical frames in one frozen pose.
- Change hand height, distance, and angle every 25-40 samples.
- Use both front-camera style and back-camera style photos.
- Keep bad samples out. If your hand was half out of frame, skip and recollect.
- After retraining, test before collecting more. Bad balance can make the model worse.

## When To Stop

Stop collecting a sign when it has enough samples and passes manual tests in:

- Live mode.
- Image mode with front-camera and back-camera photos.
- Normal light, dim light, and bright light.
- At least two distances from the camera.

If a sign still fails after it has many samples, collect more variety, not just more of the same pose.

## Project Methodology

This methodology explains the approach used to research, build, train, and test SMART SIGN.

### Research Approach

The project started by researching how sign language recognition systems usually work. The main idea found from research is that computer vision can be used to detect hand landmarks, and machine learning can then classify those landmarks into signs. Instead of training directly on full images, SMART SIGN uses hand landmark points because they are smaller, faster to process, and easier for a model to learn from.

The project also researched assistive communication tools, speech-to-text, text-to-speech, and avatar-based signing. This helped shape SMART SIGN as more than just a sign detector. The goal became a communication tool that can support sign-to-text, voice-to-sign, emergency phrases, image translation, and call-based communication.

Main research areas:

- Hand tracking and landmark detection.
- American Sign Language alphabet and common word signs.
- Machine learning classification for static signs.
- Motion recognition for moving signs like `J`, `Z`, and word signs.
- Text-to-speech and speech-to-text for two-way communication.
- Web camera access, phone testing, and browser-based deployment.

### Tools And Technologies Used

SMART SIGN was built using Python and web technologies. Python was used for the backend, data collection, model training, and prediction logic. The web interface was used so the system could run through a browser using a webcam.

Key tools used:

- **Python**: main programming language for training, recognition, and backend logic.
- **Flask**: web framework for running the app.
- **MediaPipe**: used for hand landmark detection.
- **scikit-learn**: used for static sign and gesture classification models.
- **PyTorch/LSTM**: used for sequence-based motion gesture recognition.
- **OpenCV**: used for camera frame processing during data collection and recognition.
- **Text-to-speech and speech-to-text tools**: used for voice-based communication features.
- **HTML, CSS, and JavaScript**: used for the frontend interface.

### Data Collection Method

Training data was collected using the project’s own data collection scripts. Static alphabet signs were collected with `data_collector.py`, while motion signs and word signs were collected with `gesture_collector.py`.

For static signs, the system records hand landmark positions from the camera and saves them with the correct label, such as `A`, `B`, or `C`. For motion signs, the system records a short sequence of hand landmarks across multiple frames so the model can learn movement over time.

The data collection method focused on variety, not just quantity. Samples were collected under different conditions so the model could work better in real use.

Collection conditions included:

- Normal room lighting.
- Dim lighting.
- Bright or window lighting.
- Different distances from the camera.
- Slight left and right body angles.
- Front-camera and back-camera style positions.
- Different signing speeds for motion signs.
- Different backgrounds where possible.

Bad samples were avoided or recollected. Examples of bad samples include hands being cut off, blurry frames, wrong labels, or repeated samples where the hand did not move naturally between recordings.

### Training Method

After collecting data, the models were trained using the project’s training scripts.

Static alphabet recognition was trained using:

`python model_trainer.py`

Motion and word sign recognition were trained using:

`python gesture_trainer.py`
`python lstm_gesture_trainer.py`

The static model learns from single hand poses. The motion models learn from gesture sequences, which are needed for signs that involve movement. The LSTM model was included because it is better suited for time-based data, where the order of frames matters.

The training process included:

- Loading saved landmark data.
- Splitting data into training and validation sets.
- Applying data augmentation where useful.
- Training the model.
- Saving the trained model files.
- Saving model metrics for later review.

### Testing And Evaluation Method

The project was tested using both automatic reports and manual testing. Automatic reports helped check sample counts and model accuracy. Manual testing was needed because high model accuracy does not always mean the system works perfectly in real camera conditions.

Testing methods included:

- Running `python tools/daily_accuracy_report.py`.
- Running `python _system_audit.py`.
- Testing letters A-Z in Live mode.
- Testing motion signs such as `J`, `Z`, `HELP`, `STOP`, and `WATER`.
- Uploading photos in Image mode.
- Testing speech-to-text and text-to-speech.
- Testing Call mode with one user signing and another receiving speech or text.

The model was evaluated based on:

- Prediction accuracy.
- Confidence score.
- Whether the correct sign appeared in Live mode.
- Whether the app rejected unclear or low-confidence signs.
- Whether Image mode worked with different photo types.
- Whether the communication flow worked in a realistic demo.

### Ethical And Practical Considerations

Because the project uses camera input and training samples, privacy is important. The project should avoid saving unnecessary personal images and should clearly explain what data is collected. Since sign language communication can be important in real-life situations, the system should also be honest about its limitations. It should not claim perfect translation or replace a professional interpreter.

The project is best understood as a prototype assistive communication tool. It can support basic communication and learning, but it still needs more testing with real users before being treated as a fully reliable accessibility system.

### Limitations Of The Method

The main limitation is that much of the training data may come from a small number of users. This can make the model perform well during testing but less accurately for new users. Lighting, camera quality, hand size, skin tone, background, and signing style can all affect recognition.

Another limitation is that the system currently recognizes a limited set of letters and word signs. Full sign language translation is much more complex because sign language includes grammar, facial expression, body movement, and context.

To improve the method, future work should collect data from more users, add more signs, improve motion recognition, and test the system in more realistic environments.

## Future Implementation Plan: 6 Weeks

This section separates normal improvements from new future implementations. SMART SIGN already has Live recognition, Image mode, Speech mode, Call mode, training scripts, and avatar signing. The future work should improve those parts, but it should also add a few realistic new features that make the project feel more complete.

### Existing Features That Still Need Improvement

- Training: collect more real-world samples and validate the high model scores with different people, phones, rooms, skin tones, lighting, and signing speeds.
- Motion recognition: make the LSTM gesture model the main model for moving signs like `J`, `Z`, `HELP`, and `STOP`.
- Image mode: test with more phone photos, back-camera photos, low light, and rotated hands.
- Call mode: improve reliability and document the limitation that call signaling is currently stored in memory.
- Avatar signing: verify more word animations and improve fallback behavior when a word has no matching sign.
- Deployment: test the app on HTTPS with camera and microphone permissions enabled.

### New Future Implementations

These are the new things that could be added within 6 weeks.

1. Practice Mode

Practice Mode would let users choose a letter or word, copy an example sign, and get instant feedback from the camera. Instead of only translating signs, the system would also tell the user whether they are practicing correctly. It would show the expected sign, the detected sign, the confidence score, and a simple result such as "correct", "try again", or "move your hand into frame".

2. Training Feedback Dashboard

The dashboard would show which signs have enough samples and which signs still need more data. It would include sample counts, weak labels, latest accuracy, and failed manual tests. This would make training easier because the user would not have to read raw files or remember which signs were weak.

3. Custom Phrase Library

The phrase library would let users save common phrases such as "I need help", "I want water", "thank you", or "please call someone". These phrases could be used in Speech mode, avatar signing, and emergency communication. This is realistic because the project already has emergency phrases and text-to-speech, so the new work would be organizing and saving user phrases.

4. Personal Settings Profile

This would let a user save settings such as preferred voice, speech speed, confidence threshold, camera preference, and favorite phrases. It would make the app feel more personal and reduce setup time each time the user opens it.

5. Mistake Review System

When the model gets a sign wrong, the user could mark it as incorrect. The app would save the expected sign, predicted sign, confidence score, and timestamp. Later, these mistakes could be used to decide what to retrain. This is a new feature that connects normal use back into the training process.

6. Privacy And Data Export Controls

This would add a small privacy section where users can see what data is saved, clear local training samples, and export model reports or practice history. Since the app uses camera input and training data, this would make the project more responsible and easier to explain.

### 6 Week Timeline

### Week 1: Training Dataset Audit And Baseline

Goal: clean up the current training data before adding new features.

- Run `python tools/daily_accuracy_report.py` and record current sample counts.
- Remove bad samples where the hand is cut off, blurred, mislabeled, or duplicated.
- Save baseline model metrics before new training starts.
- List the weakest static letters and motion signs.

New implementation started: create the data structure for the Mistake Review System.

Expected outcome: a clean baseline and a place to start saving recognition mistakes.

### Week 2: Practice Mode

Goal: add a new learning feature where users can practice signs.

- Build a Practice Mode screen with selectable letters and common words.
- Show the expected sign, detected sign, and confidence score.
- Add simple feedback messages such as "correct", "try again", and "no hand detected".
- Connect Practice Mode to the existing Live recognition model.

Expected outcome: users can practice signs instead of only translating them.

### Week 3: Training Feedback Dashboard

Goal: make the training process easier to understand.

- Show sample counts for each static and motion sign.
- Highlight signs below the target sample count.
- Show latest model accuracy from saved metrics files.
- Display mistakes saved by the Mistake Review System.

Expected outcome: the project has a visible training dashboard instead of only command-line reports.

### Week 4: Custom Phrase Library And Settings Profile

Goal: add personalization.

- Let users save, edit, and delete common phrases.
- Connect saved phrases to Speech mode and emergency communication.
- Save basic user preferences such as voice, speech speed, and confidence threshold.
- Store settings locally first so the feature stays realistic for the timeline.

Expected outcome: users can personalize the app without needing a full account system yet.

### Week 5: Motion Training And Mistake Review

Goal: use real mistakes to guide training.

- Collect more samples for weak motion signs like `J`, `Z`, `HELP`, `STOP`, and `WATER`.
- Retrain with `python gesture_trainer.py` and `python lstm_gesture_trainer.py`.
- Review saved mistakes and recollect signs that fail often.
- Compare the old gesture model with the LSTM model and document which one should be used.

Expected outcome: motion recognition improves and training decisions are based on recorded failures.

### Week 6: Privacy Controls, Deployment, And Final Documentation

Goal: prepare the project for presentation or handoff.

- Add a privacy/data section explaining saved samples, reports, and settings.
- Add buttons or instructions for clearing local saved practice history.
- Test deployment on HTTPS with phone camera and microphone permissions.
- Run `_system_audit.py` and fix only high-impact issues.
- Document completed features, unfinished areas, and future recommendations.

Expected outcome: a stronger final version with new features, honest limitations, and clearer documentation.

### Future Work After 6 Weeks

- Move call signaling from in-memory storage to Redis or another shared service.
- Add real user accounts with a database.
- Add more verified avatar sign animations.
- Add a full confusion-matrix report for model training.
- Expand the phrase library into categories such as health, school, travel, and emergency.

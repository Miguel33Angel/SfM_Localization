# SfM_Localization

Basic SfM program that calculates the position of a camera from a video. For now it uses ORB detectors, brute force NNDR and Ransac to get matches and thus the Fundamental matrix between 2 frames.
There's plans to substitute the keypoint detector and matches with SuperGlue+SuperPoint or the Key.Net Keypoint detector with other matcher.

The videos are from an assigment, but I plan to get videos from flying drones, to test them in big space areas, where points are further away and it's more difficult.

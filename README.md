# SfM_Localization

Basic SfM program that calculates the position of a camera from a video. For now it uses SuperGlue+SuperPoint and Ransac to get matches and their inliers and thus the Fundamental matrix between 2 frames (P8P).
There's plans to substitute the keypoint detector and matcher with the Key.Net Keypoint detector and other state of the art DL matchers, as SP and SG appear to be worse not state of the art in most datasets.

The videos are from an assigment, but I plan to get videos from flying drones, to test them in big space areas, where points are further away and it's more difficult to do SfM.

# This is the Calibration used during the data recording All sequences work with
# this calibration values. The results are obtained with Kalibr and are given in
# yaml, txt and pdf format.  Those are essentialy the same results as in
# 00_calib. No big difference since the beamsplitter setup did not change.

# However, this results contains the IMU calibration. This is the transformation
# T_cam0_imu (cam0 is the RGB camera) and T_cam1_imu (cam1 is the event camera)
# These transformations were computed using the 01_hand_eye_vrpn sequences. the
# same sequence used for the hand-eye calibration in the RPG Flying room. The numbers
# are given by the Kalibr tool. We can trust the rotation values which are correct. The
# translation seems too big we know the IMU is the Prophesse IMU which is
# located at the event camera. We could neglect the translation values of the
# transformation and keep the rotation.



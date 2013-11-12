std_msgs/Header header
# Tracking info
uint32 ActiveTracks
uint32 TotalTracks
person_tracker/TrackedPerson[] Tracks
# Tracker info
uint8 POSEONLY=1
uint8 SKELETON=2
uint8 MOCAP=3
uint8 TrackerType
string TrackerName

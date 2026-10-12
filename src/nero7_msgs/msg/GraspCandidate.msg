# A grasp candidate from GR-ConvNet, already expressed in base_link frame.
builtin_interfaces/Time stamp   # Image stamp the inference came from

geometry_msgs/Point center      # Grasp point in base_link frame (m), at block top height
float64 yaw                     # Gripper yaw in base_link frame (rad), foldable to [-pi/2, pi/2)
float64 width_mm                # Recommended gripper opening in mm (grasp width x 0.85 applied here)
float64 score                   # Grasp quality score in [0, 1]
float64 u_px                    # Grasp point u in color image grid (for debug)
float64 v_px                    # Grasp point v in color image grid (for debug)

int32 status                    # 0=ok, 1=no-peak-fallback (centroid+yaw=velocity dir), 2=inference-failed

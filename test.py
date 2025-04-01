import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
print("✅ 相機啟動成功")
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
print("✅ 拿到第一幀:", color_frame is not None)
pipeline.stop()

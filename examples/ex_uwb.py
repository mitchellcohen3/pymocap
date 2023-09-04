from pymocap import RangeData, Tag, MocapTrajectory
import rosbag
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# TODO: add the correction
# TODO: test the navlie export


tags = [
    Tag(1, "ifo001", [0.131, -0.172, -0.052], -0.3960),
    Tag(2, "ifo001", [-0.175, 0.157, -0.053], 0.0574),
    Tag(3, "ifo002", [0.165, -0.150, -0.034], -0.0786),
    Tag(4, "ifo002", [-0.154, 0.169, -0.017], -0.4913),
    Tag(5, "ifo003", [0.166, -0.181, -0.055], -0.3411),
    Tag(6, "ifo003", [-0.134, 0.154, -0.051], -0.3577),
    Tag(7, "ifo001", [-0.175, 0.157, -0.053], 0.2377),
]

filename = "data/bias_calib2.bag"
agent = "ifo001"
agent_list = ["ifo001", "ifo002", "ifo003"]


# Extract data
with rosbag.Bag(filename) as bag:
    mocaps = [MocapTrajectory.from_bag(bag, a) for a in agent_list]
    uwb = RangeData.from_bag(bag, f"/{agent}/uwb/range")

# Do power-correlated bias correction
uwb_calib = uwb.apply_calibration(tags)
uwb_calib = uwb_calib.remove_outliers(mocaps, tags, max_error = 0.5)



fig, axs = uwb.plot(mocaps, tags)
fig, axs = uwb_calib.plot(mocaps, tags)
plt.show()

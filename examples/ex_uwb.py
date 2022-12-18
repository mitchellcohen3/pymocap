from pymocap import RangeData, Tag, MocapTrajectory
import rosbag
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

tags = [
    Tag(1, "ifo001", [ 0.131, -0.172, -0.052]),
    Tag(2, "ifo001", [-0.175,  0.157, -0.053]),
    Tag(3, "ifo002", [ 0.165, -0.150, -0.034]),
    Tag(4, "ifo002", [-0.154,  0.169, -0.017]),
    Tag(5, "ifo003", [ 0.166, -0.181, -0.055]),
    Tag(6, "ifo003", [-0.134,  0.154, -0.051]),
    Tag(7, "ifo003", [-0.175,  0.157, -0.053]),
]

filename = "data/bias_calib2.bag"
agent="ifo001"
agents = ["ifo001", "ifo002", "ifo003"]
with rosbag.Bag(filename) as bag:
    mocaps = [MocapTrajectory.from_bag(bag, a) for a in agents]
    uwb = RangeData.from_bag(bag, f"/{agent}/uwb/range")

uwb.plot(mocaps, tags)
plt.show()
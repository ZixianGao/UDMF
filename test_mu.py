from main_mib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# color_map = ['r','y','k','g','b','m','c']
color_map = ['r','y']
def plot_embedding_2D(data, label, title,description):

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    x_range = plt.gca().get_xlim()
    y_range = plt.gca().get_ylim()

    x_center = (x_range[0] + x_range[1]) / 2
    y_bottom = y_range[0]
    fig = plt.figure()
    for i in range(data.shape[0]):
        i=int(i)
        plt.plot(data[i, 0], data[i, 1],marker='o',markersize=1,color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.text(x_center, y_bottom - 0.1, description, fontsize=12, ha='center')

    return fig

with open(f"./datasets/{args.dataset}.pkl", "rb") as handle:
    data = pickle.load(handle)

test_data = data["test"]
test_dataset = get_appropriate_dataset(test_data)
test_data_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

model=torch.load("./save/model.pth")

test_acc2, test_mae, test_corr, test_f_score, test_acc7,vision_mus,labels = test_score_model(model, test_data_loader)
print(
            "best mae:{:.4f}, acc:{:.4f}, acc7:{:.4f}, f1:{:.4f}, corr:{:.4f}".format(
                test_mae, test_acc2, test_acc7, test_f_score, test_corr
            )
        )
labels=labels.astype(int)
tsne_2D = TSNE(n_components=3, init='pca', random_state=0) 
result_2D = tsne_2D.fit_transform(vision_mus)
    
print('Finished......')
fig1 = plot_embedding_2D(result_2D, labels, 't-SNE','ours-UE+CA+IB')
fig1.show()
plt.pause(50)

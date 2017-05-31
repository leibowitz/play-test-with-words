from surprise import SVD
from surprise import Dataset
from surprise import KNNBasic
from surprise import evaluate, print_perf


# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
# Build an algorithm, and train it.
algo = KNNBasic()
algo.train(trainset)


uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)

# and split it into 3 folds for cross-validation.
data.split(n_folds=3)
perf = evaluate(algo, data)
print_perf(perf)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)

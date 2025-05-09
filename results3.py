from torch.utils.data import DataLoader, ConcatDataset

from test import load_data, train_test_cycle

def run_all_train_test_combinations(train_datasets, test_datasets, model_fn, batch_size, lr, epochs, threshold):
    from itertools import combinations, product

    results = []

    # Generate all non-empty combinations of train and test datasets
    train_combos = []
    for i in range(1, len(train_datasets)+1):
        train_combos.extend(combinations(train_datasets, i))

    test_combos = []
    for i in range(1, len(test_datasets)+1):
        test_combos.extend(combinations(test_datasets, i))

    for train_set, test_set in product(train_combos, test_combos):
        # Create new model instance each time
        model = model_fn()

        # Create loaders
        train_loader = DataLoader(ConcatDataset(train_set), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(ConcatDataset(test_set), batch_size=batch_size, shuffle=True)

        # Train and evaluate
        f1_scores = train_test_cycle((train_loader, test_loader), model,
                                     batch_size=batch_size, lr=lr,
                                     epochs=epochs, threshold=threshold)

        max_f1 = max(f1_scores)
        train_names = '+'.join([ds.name for ds in train_set])  # assumes datasets have a `.name` attribute
        test_names = '+'.join([ds.name for ds in test_set])
        print(f"Train: {train_names} | Test: {test_names} | Max F1: {max_f1:.4f}")

        results.append(((train_names, test_names), max_f1))

    return results

batch_size=64

train_1, test_1 = load_data(challenge=True)
train_2, test_2 = load_data(challenge=False)

# Create DataLoaders
train_loader = DataLoader(ConcatDataset([train_1]), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ConcatDataset([test_1]), batch_size=batch_size, shuffle=True)



train_1.name = 'train_1'
train_2.name = 'train_2'
test_1.name = 'test_1'
test_2.name = 'test_2'

results = run_all_train_test_combinations(
    train_datasets=[train_1, train_2],
    test_datasets=[test_1, test_2],
    model_fn=lambda: FetalQRSWindowDetector(),
    batch_size=32,
    lr=0.001,
    epochs=10,
    threshold=0.5
)
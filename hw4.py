import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import random
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    try:
        if os.path.exists('train_X.npy') and os.path.exists('train_y.npy'):
            train_X = np.load('train_X.npy', allow_pickle=True)
            train_y = np.load('train_y.npy', allow_pickle=True)
            test_X = np.load('test_X.npy', allow_pickle=True)
            test_y = np.load('test_y.npy', allow_pickle=True)
        else:
            train_data_obj = np.load('train_data.npy', allow_pickle=True)
            test_data_obj = np.load('test_data.npy', allow_pickle=True)
            
            def extract_flattened_data(data_obj):
                if data_obj.ndim == 0: data_obj = data_obj.item()
                all_images, all_labels = [], []
                iterator = data_obj.values() if isinstance(data_obj, dict) else data_obj
                for client in iterator:
                    if isinstance(client, dict) and 'images' in client:
                        all_images.append(client['images'])
                        all_labels.append(client['labels'])
                if not all_images: raise ValueError("No data found.")
                return np.concatenate(all_images), np.concatenate(all_labels)

            train_X, train_y = extract_flattened_data(train_data_obj)
            test_X, test_y = extract_flattened_data(test_data_obj)

        print(f"Original Dataset Size: {len(train_y)}")
        train_mask = train_y < 10
        train_X = train_X[train_mask]
        train_y = train_y[train_mask]
        
        test_mask = test_y < 10
        test_X = test_X[test_mask]
        test_y = test_y[test_mask]
        print(f"Filtered Dataset Size (Digits Only): {len(train_y)}")

        unique_labels = np.unique(train_y)
        num_classes = len(unique_labels)
        label_map = {val: i for i, val in enumerate(np.sort(unique_labels))}
        train_y = np.array([label_map[y] for y in train_y])
        test_y = np.array([label_map[y] for y in test_y if y in label_map]) 

        if train_X.max() > 1.0:
            train_X = train_X.astype(np.float32) / 255.0
            test_X = test_X.astype(np.float32) / 255.0
        else:
            train_X = train_X.astype(np.float32)
            test_X = test_X.astype(np.float32)

        if len(train_X.shape) > 2:
            train_X = train_X.reshape(train_X.shape[0], -1)
            test_X = test_X.reshape(test_X.shape[0], -1)

        train_dataset_full = TensorDataset(torch.tensor(train_X), torch.tensor(train_y).long())
        test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_y).long())

        train_size = int(0.8 * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        train_subset, val_subset = random_split(train_dataset_full, [train_size, val_size])

        return train_subset, val_subset, test_dataset, train_X.shape[1], num_classes

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_name):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        if activation_name.lower() == 'relu': self.activation = nn.ReLU()
        elif activation_name.lower() == 'sigmoid': self.activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh': self.activation = nn.Tanh()
        else: self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))

def train_evaluate(batch_size, activation_name, train_data, val_data, input_dim, num_classes, epochs=5):
    bs = int(max(16, min(1024, batch_size)))
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
    
    model = SimpleClassifier(input_dim, 256, num_classes, activation_name).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            outputs = model(X_b)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
            
    return f1_score(all_labels, all_preds, average='macro')

class GeneticAlgorithm:
    def __init__(self, pop_size, generations, train_data, val_data, input_dim, num_classes):
        self.pop_size = pop_size
        self.generations = generations
        self.train_data = train_data
        self.val_data = val_data
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.activations = ['relu', 'sigmoid', 'tanh']
        self.population = [] 
        
    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            self.population.append({
                'bs': random.randint(16, 1024),
                'act': random.choice(self.activations),
                'fitness': -1
            })

    def roulette_selection(self):
        total_fit = sum(ind['fitness'] for ind in self.population)
        if total_fit == 0: return random.choice(self.population)
        pick = random.uniform(0, total_fit)
        current = 0
        for ind in self.population:
            current += ind['fitness']
            if current > pick: return ind
        return self.population[-1]

    def run(self):
        self.initialize()
        
        for ind in self.population:
            ind['fitness'] = train_evaluate(ind['bs'], ind['act'], self.train_data, self.val_data, self.input_dim, self.num_classes, epochs=10)
        
        avg_history, best_history = [], []

        for gen in range(self.generations):
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            best_fitness = self.population[0]['fitness']
            avg_fitness = np.mean([i['fitness'] for i in self.population])
            best_history.append(best_fitness)
            avg_history.append(avg_fitness)
            
            print(f"GA Gen {gen+1}/{self.generations}: Best F1={best_fitness:.4f}")
            
            new_pop = [self.population[0]] 
            while len(new_pop) < self.pop_size:
                p1 = self.roulette_selection()
                p2 = self.roulette_selection()
                c = {'bs': p1['bs'], 'act': p2['act'], 'fitness': -1}
                if random.random() < 0.15: c['bs'] = random.randint(16, 1024)
                if random.random() < 0.15: c['act'] = random.choice(self.activations)
                
                c['fitness'] = train_evaluate(c['bs'], c['act'], self.train_data, self.val_data, self.input_dim, self.num_classes, epochs=10)
                new_pop.append(c)
            self.population = new_pop

        plt.figure()
        plt.plot(range(1, self.generations+1), avg_history, label='Avg Fitness')
        plt.plot(range(1, self.generations+1), best_history, label='Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Validation F1 Score')
        plt.title('Genetic Algorithm Progress')
        plt.legend()
        plt.savefig('ga_progress.png')
        
        self.population.sort(key=lambda x: x['fitness'], reverse=True)
        return self.population[0]

def run_bayesian_optimization(train_data, val_data, input_dim, num_classes):
    def bo_target(batch_size, activation_idx):
        act_list = ['relu', 'sigmoid', 'tanh']
        idx = max(0, min(2, int(round(activation_idx))))
        return train_evaluate(batch_size, act_list[idx], train_data, val_data, input_dim, num_classes, epochs=10)

    optimizer = BayesianOptimization(
        f=bo_target,
        pbounds={'batch_size': (16, 1024), 'activation_idx': (0, 2)},
        random_state=SEED,
        verbose=2
    )
    optimizer.maximize(init_points=10, n_iter=40)
    p = optimizer.max['params']
    return int(p['batch_size']), ['relu', 'sigmoid', 'tanh'][int(round(p['activation_idx']))]

def final_train_and_report(bs, act, train_data, val_data, test_data, input_dim, num_classes, method_name):
    print(f"\n--- Final Training ({method_name}) with BS={bs}, Act={act} ---")
    combined = torch.utils.data.ConcatDataset([train_data, val_data])
    loader = DataLoader(combined, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
    
    model = SimpleClassifier(input_dim, 256, num_classes, act).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    train_f1_history = []
    epochs = 50 
    
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            _, p = torch.max(outputs, 1)
            all_preds.extend(p.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
        f1 = f1_score(all_labels, all_preds, average='macro')
        train_f1_history.append(f1)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train F1 = {f1:.4f}")

    plt.figure()
    plt.plot(range(1, epochs+1), train_f1_history, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Training F1 Score')
    plt.title(f'Training Curve ({method_name})')
    plt.savefig(f'training_curve_{method_name}.png')

    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            _, p = torch.max(model(X), 1)
            all_p.extend(p.cpu().numpy())
            all_y.extend(y.cpu().numpy())
            
    test_f1 = f1_score(all_y, all_p, average='macro')
    print(f"Final Test F1 Score ({method_name}): {test_f1:.4f}")
    return test_f1

if __name__ == "__main__":
    train_sub, val_sub, test_set, dim, n_classes = load_data()
    if train_sub:
        ga = GeneticAlgorithm(pop_size=20, generations=20, train_data=train_sub, val_data=val_sub, input_dim=dim, num_classes=n_classes)
        best_ga = ga.run()
        print(f"GA Selected: BS={best_ga['bs']}, Act={best_ga['act']}")
        
        bo_bs, bo_act = run_bayesian_optimization(train_sub, val_sub, dim, n_classes)
        print(f"BO Selected: BS={bo_bs}, Act={bo_act}")
        
        ga_test_f1 = final_train_and_report(best_ga['bs'], best_ga['act'], train_sub, val_sub, test_set, dim, n_classes, "GA")
        bo_test_f1 = final_train_and_report(bo_bs, bo_act, train_sub, val_sub, test_set, dim, n_classes, "BO")
        
        print("\n--- Final Results Summary ---")
        print(f"Genetic Algorithm Test F1: {ga_test_f1:.4f}")
        print(f"Bayesian Optimization Test F1: {bo_test_f1:.4f}")
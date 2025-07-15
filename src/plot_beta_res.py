import matplotlib.pyplot as plt
import csv

csv_path = "results/beta_results.csv"

def load_csv():
    with open(csv_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = list(reader)
    return data

# csv format:
# Optimizer, beta, num_params, cost

# plot beta vs num_params
def plot_beta_vs_num_params(data):
    betas = [float(row[1]) for row in data]
    num_params = [int(row[2]) for row in data]

    plt.figure(figsize=(10, 6))
    plt.scatter(betas, num_params, marker='o')
    plt.title('Beta vs Number of Parameters')
    plt.xlabel('Beta')
    # xlog scale
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylabel('Number of Parameters')
    plt.grid()
    plt.savefig('beta_vs_num_params.png')
    plt.show()
    print("Beta vs Number of Parameters plot saved as 'beta_vs_num_params.png'.")
    
# plot beta vs cost
def plot_beta_vs_cost(data):
    betas = [float(row[1]) for row in data]
    costs = [float(row[3]) for row in data]

    plt.figure(figsize=(10, 6))
    plt.scatter(betas, costs, marker='o')
    plt.title('Beta vs Cost')
    plt.xlabel('Beta')
    plt.xscale('log')
    plt.ylabel('Cost')
    plt.grid()
    plt.savefig('beta_vs_cost.png')
    plt.show()
    print("Beta vs Cost plot saved as 'beta_vs_cost.png'.")
    
def main():
    data = load_csv()
    # print(data)
    # filter names that arent "FracOptimizer"
    data = [row for row in data if row[0] == "FracOptimizer"]

    plot_beta_vs_num_params(data)
    plot_beta_vs_cost(data)
    
if __name__ == "__main__":
    main()
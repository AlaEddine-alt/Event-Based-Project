
def write_results_to_file(method, best_accuracy, time, filename="results.txt"):
    with open(filename, "a", newline="") as f:
        f.write("--------------------------------\n")
        f.write(f"{method}\n")
        f.write(f"Test Accuracy: {best_accuracy:.2f}%\n")
        f.write(f"Training + evaluation time: {time:.2f} seconds\n")

def write_filtering_results_to_file(method, err, time, filename="filtering_results.txt"):
    with open(filename, "a", newline="") as f:
        f.write("--------------------------------\n")
        f.write(f"{method}\n")
        f.write(f"Average Filtering Error (ERR): {err:.4f}\n")
        f.write(f"Filtering time: {time:.2f} seconds\n")

def write_parameter_tuning_results_to_file(method, param_value, err, time, acc, train_time, filename="tuning_results.txt"):
    with open(filename, "a", newline="") as f:
        f.write("--------------------------------\n")
        f.write(f"{method} - Parameter: {param_value}\n")
        f.write(f"Average Event Reduction Ratio (ERR): {err:.4f}\n")
        f.write(f"Filtering time: {time:.2f} seconds\n")
        f.write(f"Test Accuracy: {acc:.2f}%\n")
        f.write(f"Training time: {train_time:.2f} seconds\n")

import os, io, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


ROOT_DATA = "data"
MAX_dy = 350
MIN_dy = 100
STEP_SIZE = 10

def data_loader(fname: str):
    lines = None
    with open(os.path.join(ROOT_DATA, fname), "r") as f:
        lines = f.readlines()
    
    if not lines:
        print(f"[ERR] {fname} could not be read")
        return None
    
    # skip empty lines
    lines = [line.strip() for line in lines if line.strip()]
    print(f"[INF] validating lines at 0 and 1 index: {lines[0]} and {lines[1]}")

    # parse lines
    result_str_t = []
    result_t = []
    result_y = []
    ignored_c = 0
    for line in lines:
        splt = line.split(";")
        #print(splt)
        try:
            result_t.append(
                float(splt[0].replace(",", "."))
            )
            result_str_t.append(splt[0])
            result_y.append(
                int(splt[1])
            )
        except:
            ignored_c += 1
    
    if ignored_c > 0:
        print(f"[WAR] ignored lines: {ignored_c}")
    print(f"[INF] done loading dataset, arr len {len(result_t)}")
    return result_t, result_y, result_str_t


def check_dt_consistency(t_vals, desired_dt=0.1):
    for i in range(1, len(t_vals)):
        if round(t_vals[i] - t_vals[i-1], 10) != desired_dt:
            print(f"[ERR] inconsistency found at {i+1}")
            print(f"[VAL] {t_vals[i]} - {t_vals[i-1]} = {t_vals[i] - t_vals[i-1]}")
            return False

    print("[INF] consistency check passed")
    return True


def rate_limit(x, dt=0.1, dy_cap=None):
    x = np.asarray(x, dtype=float)
    y = x.copy()

    if not dy_cap:
        print("[WAR] dy cap not given, setting global")
        dy_cap = MAX_dy

    # forward pass
    for i in range(1, len(x)):
        up = y[i-1] + dy_cap
        dn = y[i-1] - dy_cap
        y[i] = min(max(x[i], dn), up)

    # backward pass
    for i in range(len(x)-2, -1, -1):
        up = y[i+1] + dy_cap
        dn = y[i+1] - dy_cap
        y[i] = min(max(y[i], dn), up)

    return [int(r) for r in y]


def adaptive_damping(values, base_alpha=0.6, strong_alpha=0.4, threshold=0.5, scale=0.2):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    
    for i in range(1, len(values)):
        diff = values[i] - values[i-1]
        excess = max(0.0, (abs(diff) - threshold) / scale)
        mix = 1 / (1 + np.exp(-excess)) # sigmoid
        alpha = base_alpha * (1 - mix) + strong_alpha * mix
        out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
    
    return out


def validate_rate_limited_dataset(dset, dy_cap=None):
    if not dy_cap:
        print("[WAR] dy cap not given, setting global")
        dy_cap = MAX_dy
        
    for i in range(1, len(dset)):
        if abs(dset[i] - dset[i-1]) > dy_cap:
            print("[ERR] rate limited dataset still has over the max limit diff")
            return False
    print("[INF] rate limited dataset capped dy passed all checks")
    return True


if __name__=="__main__":
    if not os.path.isdir("bulk_results/"):
        print("[INF] creating results dir")
        os.mkdir("bulk_results")

    files = os.listdir(ROOT_DATA)
    steps = [MIN_dy+i*STEP_SIZE for i in range(0, int((MAX_dy-MIN_dy)/STEP_SIZE)+1)]
    strong_alphas = [i/10 for i in range(1, 5)]
    # print(steps)
    # sys.exit(0)
    for dy_cap in tqdm(steps):
        os.mkdir(os.path.join("bulk_results/", f"max_abs_rpm_limit_{str(dy_cap)}"))
        for strong_alpha in strong_alphas:
            path = os.path.join("bulk_results/", f"max_abs_rpm_limit_{str(dy_cap)}/strong_alpha_{str(strong_alpha)}")
            os.mkdir(path)

            for file in files:
                print(f"working on {file}...")
                t_arr, y_arr, t_str_arr = data_loader(fname=file)
                
                if not check_dt_consistency(t_vals=t_arr):
                    print(f"[INF] fix and rerun script, skipping {file}")
                    continue
                
                result = adaptive_damping(
                    rate_limit(y_arr, dy_cap=dy_cap),
                    strong_alpha=strong_alpha
                )

                if not validate_rate_limited_dataset(dset=result, dy_cap=dy_cap):
                    print(f"[INF] skipping file due to over the limit limiting err {file}")
                    continue

                # Plot original vs adjusted
                plt.figure(figsize=(8, 4))
                plt.plot(t_arr, y_arr, label="Original RPM", marker="o")
                plt.plot(t_arr, result, label="Rate-Limited RPM", marker="s")
                plt.xlabel("Time (s)")
                plt.ylabel("RPM")
                plt.title(f"Original vs Rate-Limited RPM - {file.split(".")[0]}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{path}/{file.split(".")[0]}.png", dpi=300, bbox_inches="tight")
                #plt.show()

                f = open(f"{path}/{file.split(".")[0]}_rate_limited.csv", "w")
                for i in range(0, len(result)):
                    f.write(
                        str(t_str_arr[i]) + ";" + str(result[i]) + "\n\n"
                    )
                f.close()

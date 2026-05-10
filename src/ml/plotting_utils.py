import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = 'diploma_writing/ml_schedules/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_scatter(actual, predicted, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.5, color='blue', s=10)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.xlabel('Реальное $\log_{10}(\mu)$')
    plt.ylabel('Предсказанное $\log_{10}(\mu)$')
    #plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_residuals(actual, predicted, title, filename):
    residuals = actual - predicted
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel('Ошибка ($\log_{10}(\mu)_{true} - \log_{10}(\mu)_{pred}$)')
    plt.ylabel('Частота')
    #plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_r2_vs_k(k_values, r2_values, title, filename):
    """
    Plots the actual R2 vs k relationship.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, r2_values, marker='o', linestyle='-', color='darkblue', linewidth=2, markersize=8)
    plt.xlabel('Количество датчиков (k)')
    plt.ylabel('$R^2$')
    #plt.title(title)
    plt.xticks(k_values)
    plt.ylim(0.8, 0.85)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


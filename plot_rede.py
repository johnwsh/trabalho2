import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_network_performance():
    if not os.path.exists('confusion_matrix_rede.csv') or not os.path.exists('metrics_rede.csv'):
        print("[-] Erro: Arquivos .csv da Rede Neural não encontrados!")
        return

    cm = np.loadtxt('confusion_matrix_rede.csv', delimiter=',')
    df = pd.read_csv('metrics_rede.csv')

    plt.rcParams['font.family'] = 'sans-serif'
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle('Dashboard: Rede Neural (Regressão vs Classificação)', fontsize=18, fontweight='bold', color='#2c3e50')

    # PLOT 1: Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(cm, cmap='Blues', interpolation='nearest') # Azul para a Rede Neural
    ax1.set_title('Matriz de Confusão (Rede de Classificação)', fontsize=13, fontweight='bold', pad=10)
    fig.colorbar(im, ax=ax1, shrink=0.8)
    
    classes = ['Classe 1', 'Classe 2', 'Classe 3', 'Classe 4']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=10)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontweight='bold', fontsize=12)
                     
    ax1.set_ylabel('Classe Real (Gabarito)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Classe Prevista pelo Neurônio Mais Forte (Argmax)', fontsize=11, fontweight='bold')

    # PLOT 2: Barras
    ax2 = fig.add_subplot(2, 2, 2)
    df_class = df[df['class'] != 'all'].copy()
    df_class['class'] = 'Classe ' + df_class['class'].astype(str)
    df_pivot = df_class.pivot(index='class', columns='metric', values='value')
    
    df_pivot[['Precision', 'Recall', 'F1-Measure']].plot(kind='bar', ax=ax2, width=0.75, color=['#3498db', '#2ecc71', '#e67e22'])
    ax2.set_title('Métricas da Rede de Classificação', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylabel('Score (0.00 a 1.00)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower left', frameon=True)
    plt.setp(ax2.get_xticklabels(), rotation=0, fontsize=10)

    # PLOT 3: Card
    ax3 = fig.add_subplot(2, 1, 2)
    ax3.axis('off')
    
    rmse_val = df[df['metric'] == 'RMSE']['value'].values[0]
    acc_val = df[df['metric'] == 'Accuracy']['value'].values[0]
    
    box_text = (
        "========================================================================\n"
        "                     MÉTRICAS K-FOLD: REDES NEURAIS INDEPENDENTES       \n"
        "========================================================================\n\n"
        f"   • ACURÁCIA (Rede de Classificação): {acc_val:.2f}%\n"
        f"   • RMSE     (Rede de Regressão):     {rmse_val:.5f}\n\n"
        "------------------------------------------------------------------------\n"
        " Metodologia: Foram treinadas DUAS Redes Neurais (MLP) distintas.       \n"
        " A rede de regressão gerou saídas contínuas avaliadas via RMSE.         \n"
        " A rede de classificação utilizou One-Hot Encoding e 4 neurônios na     \n"
        " camada de saída, sendo avaliada extraindo o neurônio de maior força.   \n"
        "========================================================================"
    )
    
    ax3.text(0.05, 0.2, box_text, fontsize=12, family='monospace',
             bbox=dict(facecolor='#f4f8fb', edgecolor='#3498db', boxstyle='round,pad=1.5'))

    plt.tight_layout()
    output_img = 'network_performance_dashboard.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"[+] Dashboard salvo como '{output_img}'")
    plt.show()

if __name__ == '__main__':
    plot_network_performance()
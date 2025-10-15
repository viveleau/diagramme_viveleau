import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
import io
import base64
from fpdf import FPDF
import tempfile
import os
from io import BytesIO
import base64

# Dictionnaire des coefficients de Hazen-Williams par type de canalisation
HAZEN_WILLIAMS_COEFFICIENTS = {
    "Acier": 130,
    "Acier galvanisé": 120,
    "Fonte": 100,
    "Fonte neuve": 130,
    "Fonte ancienne": 80,
    "PVC": 150,
    "PEHD": 140,
    "Béton": 120,
    "Cuivre": 140,
    "Fibrociment": 140
}

def convert_flow_units(flow, unit):
    """Convertit le débit entre les différentes unités"""
    if unit == "l/s":
        return flow, flow * 3.6  # l/s to m³/h
    elif unit == "m³/h":
        return flow / 3.6, flow  # m³/h to l/s
    return flow, flow

def calculate_hazen_williams(Q=None, D=None, J=None, V=None, Q_unit="m³/h", pipe_type="Acier"):
    """
    Calcule les paramètres manquants using Hazen-Williams
    Formule: Q = 0.27855 * C * D^2.63 * J^0.54
    où Q en m³/s, D en m, J en m/m
    """
    # Récupérer le coefficient C selon le type de canalisation
    C = HAZEN_WILLIAMS_COEFFICIENTS.get(pipe_type, 130)
    
    # Stocker les paramètres d'origine pour identifier ce qui a été saisi
    original_params = {'Q': Q, 'D': D, 'J': J, 'V': V, 'Q_unit': Q_unit, 'pipe_type': pipe_type}
    
    # Conversion du débit en unités standard (m³/s pour les calculs)
    if Q is not None:
        if Q_unit == "l/s":
            Q_m3s = Q / 1000  # l/s to m³/s
            Q_ls = Q
            Q_m3h = Q * 3.6
        else:  # m³/h
            Q_m3s = Q / 3600  # m³/h to m³/s
            Q_ls = Q / 3.6    # m³/h to l/s
            Q_m3h = Q
    else:
        Q_m3s = None
        Q_ls = None
        Q_m3h = None
    
    # Conversion diamètre
    if D is not None:
        D_m = D / 1000  # mm to m
    else:
        D_m = None
    
    # Compter le nombre de paramètres fournis
    provided_params = sum(1 for param in [Q, D, J, V] if param is not None)
    
    if provided_params != 2:
        return None, None, None, None, None, original_params, C
    
    # Cas 1: Q et D connus -> calcul V et J
    if Q is not None and D is not None:
        # Calcul de la vitesse
        A = np.pi * (D_m)**2 / 4
        V_calc = Q_m3s / A
        
        # Calcul de la perte de charge avec Hazen-Williams
        J_calc = (Q_m3s / (0.27855 * C * (D_m)**2.63)) ** (1/0.54)
        
        return Q_ls, Q_m3h, D, J_calc, V_calc, original_params, C
    
    # Cas 2: Q et J connus -> calcul D et V
    elif Q is not None and J is not None:
        # Calcul du diamètre
        D_m_calc = (Q_m3s / (0.27855 * C * J**0.54)) ** (1/2.63)
        D_calc = D_m_calc * 1000
        
        # Calcul de la vitesse
        A = np.pi * (D_m_calc)**2 / 4
        V_calc = Q_m3s / A
        
        return Q_ls, Q_m3h, D_calc, J, V_calc, original_params, C
    
    # Cas 3: D et J connus -> calcul Q et V
    elif D is not None and J is not None:
        # Calcul du débit
        Q_m3s_calc = 0.27855 * C * (D_m)**2.63 * J**0.54
        Q_ls_calc = Q_m3s_calc * 1000
        Q_m3h_calc = Q_m3s_calc * 3600
        
        # Calcul de la vitesse
        A = np.pi * (D_m)**2 / 4
        V_calc = Q_m3s_calc / A
        
        return Q_ls_calc, Q_m3h_calc, D, J, V_calc, original_params, C
    
    # Cas 4: V et D connus -> calcul Q et J
    elif V is not None and D is not None:
        # Calcul du débit
        A = np.pi * (D_m)**2 / 4
        Q_m3s_calc = V * A
        Q_ls_calc = Q_m3s_calc * 1000
        Q_m3h_calc = Q_m3s_calc * 3600
        
        # Calcul de la perte de charge
        J_calc = (Q_m3s_calc / (0.27855 * C * (D_m)**2.63)) ** (1/0.54)
        
        return Q_ls_calc, Q_m3h_calc, D, J_calc, V, original_params, C
    
    # Cas 5: Q et V connus -> calcul D et J
    elif Q is not None and V is not None:
        # Calcul du diamètre à partir de Q et V
        # Q = V * A = V * π * D² / 4
        # D = √(4Q / (πV))
        A = Q_m3s / V
        D_m_calc = np.sqrt(4 * A / np.pi)
        D_calc = D_m_calc * 1000
        
        # Calcul de la perte de charge avec Hazen-Williams
        J_calc = (Q_m3s / (0.27855 * C * (D_m_calc)**2.63)) ** (1/0.54)
        
        return Q_ls, Q_m3h, D_calc, J_calc, V, original_params, C
    
    # Cas 6: V et J connus -> calcul Q et D
    elif V is not None and J is not None:
        # Résolution numérique pour trouver D
        def equation(D_m):
            Q_calc = 0.27855 * C * D_m**2.63 * J**0.54
            A = np.pi * D_m**2 / 4
            V_calc = Q_calc / A
            return V_calc - V
        
        # Recherche du diamètre par dichotomie
        D_min, D_max = 0.01, 2.0  # de 10mm à 2000mm
        tolerance = 1e-6
        
        for _ in range(100):
            D_mid = (D_min + D_max) / 2
            if abs(equation(D_mid)) < tolerance:
                break
            elif equation(D_min) * equation(D_mid) < 0:
                D_max = D_mid
            else:
                D_min = D_mid
        
        D_calc = D_mid * 1000
        Q_m3s_calc = 0.27855 * C * D_mid**2.63 * J**0.54
        Q_ls_calc = Q_m3s_calc * 1000
        Q_m3h_calc = Q_m3s_calc * 3600
        
        return Q_ls_calc, Q_m3h_calc, D_calc, J, V, original_params, C
    
    else:
        return None, None, None, None, None, original_params, C

def create_dynamic_viveleau_diagram(Q_point=None, J_point=None, D_point=None, V_point=None, pipe_type="Acier"):
    """Crée un diagramme de Viveleau dynamique avec le point calculé"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Définir des échelles fixes et réalistes pour un diagramme de Viveleau
    Q_min, Q_max = 0.1, 1000  # l/s
    J_min, J_max = 0.0001, 0.1  # mCE/m
    
    # Récupérer le coefficient C selon le type de canalisation
    C = HAZEN_WILLIAMS_COEFFICIENTS.get(pipe_type, 130)
    
    # Générer des courbes précises
    Q_range_ls = np.logspace(np.log10(Q_min), np.log10(Q_max), 300)
    diameters = [50, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
    velocities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # Tracer les courbes de diamètre avec des gradients visibles
    for D in diameters:
        D_m = D / 1000
        J = (Q_range_ls/1000 / (0.27855 * C * (D_m)**2.63)) ** (1/0.54)
        
        # Filtrer les valeurs réalistes
        mask = (J > J_min) & (J < J_max) & (J > 0) & (np.isfinite(J))
        Q_filtered = Q_range_ls[mask]
        J_filtered = J[mask]
        
        if len(Q_filtered) > 2:
            # Utiliser une couleur qui varie avec le diamètre
            color_intensity = np.log(D) / np.log(1000)
            color = (0.2, 0.4, 0.8, 0.7 + 0.3 * color_intensity)
            
            ax.loglog(Q_filtered, J_filtered, color=color, linewidth=1.8, alpha=0.8)
            
            # Ajouter label pour les diamètres principaux
            if D in [100, 150, 200, 300, 400, 500]:
                # Trouver un point approprié pour le label
                mid_idx = len(Q_filtered) // 2
                if mid_idx < len(Q_filtered):
                    ax.annotate(f'DN {D}', (Q_filtered[mid_idx], J_filtered[mid_idx]), 
                               fontsize=10, alpha=0.9, 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8),
                               ha='center', va='center')
    
    # Tracer les courbes de vitesse avec un style différent
    for V in velocities:
        Q_for_V = []
        J_for_V = []
        
        for D in diameters:
            D_m = D / 1000
            A = np.pi * (D_m)**2 / 4
            Q_m3s = V * A
            Q_ls = Q_m3s * 1000
            
            if Q_min < Q_ls < Q_max:
                J_val = (Q_m3s / (0.27855 * C * (D_m)**2.63)) ** (1/0.54)
                if J_min < J_val < J_max:
                    Q_for_V.append(Q_ls)
                    J_for_V.append(J_val)
        
        if len(Q_for_V) > 1:
            # Trier par Q croissant
            sorted_indices = np.argsort(Q_for_V)
            Q_sorted = [Q_for_V[i] for i in sorted_indices]
            J_sorted = [J_for_V[i] for i in sorted_indices]
            
            # Style pour les vitesses
            ax.loglog(Q_sorted, J_sorted, 'r--', alpha=0.5, linewidth=1.2)
            
            # Ajouter label pour les vitesses principales
            if V in [1.0, 2.0, 3.0]:
                if len(Q_sorted) > 0:
                    idx = len(Q_sorted) // 2
                    ax.annotate(f'V={V} m/s', (Q_sorted[idx], J_sorted[idx]), 
                               fontsize=9, alpha=0.8, color='darkred',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='mistyrose', alpha=0.7))
    
    # Ajouter un axe secondaire pour les m³/h en haut
    ax2 = ax.twiny()
    ax2.set_xscale('log')
    
    # Définir les limites pour correspondre à la conversion CORRECTE
    Q_min_m3h = Q_min * 3.6
    Q_max_m3h = Q_max * 3.6
    ax2.set_xlim(Q_min_m3h, Q_max_m3h)
    ax2.set_xlabel('Débit (m³/h)', fontsize=12, labelpad=15)
    
    # Ajouter des lignes de repère pour m³/h (tous les 100 m³/h)
    m3h_ticks = [1, 10, 100, 1000, 3600]
    for m3h_tick in m3h_ticks:
        if Q_min_m3h <= m3h_tick <= Q_max_m3h:
            ls_tick = m3h_tick / 3.6
            ax.axvline(x=ls_tick, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
            ax2.axvline(x=m3h_tick, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    
    # Configurer les ticks de l'axe supérieur
    ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    
    # Configurer l'axe principal avec des échelles fixes
    ax.set_xlabel('Débit (l/s)', fontsize=12)
    ax.set_ylabel('Perte de charge (mCE/m)', fontsize=12)
    ax.set_title(f'Diagramme de Viveleau - {pipe_type} (C={C})', fontsize=14, pad=20)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_xlim(Q_min, Q_max)
    ax.set_ylim(J_min, J_max)
    
    # Ajouter des ticks intermédiaires pour meilleure lisibilité
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    
    # Ajouter le point calculé et les lignes de repère
    if Q_point is not None and J_point is not None:
        # Tracer le point
        ax.plot(Q_point, J_point, 'o', markersize=14, markerfacecolor='red', 
                markeredgecolor='darkred', markeredgewidth=3, label='Point de fonctionnement', zorder=10)
        
        # Tracer les lignes horizontale et verticale
        ax.axvline(x=Q_point, color='green', linestyle=':', alpha=0.7, linewidth=2, zorder=5)
        ax.axhline(y=J_point, color='green', linestyle=':', alpha=0.7, linewidth=2, zorder=5)
        
        # Annotation améliorée pour le point calculé
        annotation_text = f"Point de fonctionnement:\n"
        annotation_text += f"Débit: {Q_point:.1f} l/s\n"
        annotation_text += f"          ({Q_point*3.6:.1f} m³/h)\n"
        annotation_text += f"Perte charge: {J_point:.4f} m/m\n"
        if D_point is not None:
            annotation_text += f"Diamètre: DN {D_point:.0f} mm\n"
        if V_point is not None:
            annotation_text += f"Vitesse: {V_point:.2f} m/s"
            
        # Positionner l'annotation intelligemment
        ax.annotate(annotation_text, 
                   (Q_point, J_point), 
                   xytext=(20, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                           alpha=0.95, edgecolor='orange', linewidth=2),
                   fontsize=11, fontweight='bold', zorder=11,
                   arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        
        # Ajouter des annotations sur les axes pour les valeurs du point
        ax.annotate(f'{Q_point:.1f} l/s\n({Q_point*3.6:.1f} m³/h)', (Q_point, J_min), 
                   xytext=(0, -35), textcoords='offset points',
                   ha='center', va='top', fontsize=9, color='darkgreen',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))
        
        ax.annotate(f'{J_point:.4f} m/m', (Q_min, J_point), 
                   xytext=(-25, 0), textcoords='offset points',
                   ha='right', va='center', fontsize=10, color='darkgreen',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8))
    
    # Légende
    ax.plot([], [], 'b-', linewidth=2, label='Lignes de diamètre')
    ax.plot([], [], 'r--', linewidth=2, label='Lignes de vitesse')
    if Q_point is not None and J_point is not None:
        ax.plot([], [], 'g:', linewidth=2, label='Lignes de repère')
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # Améliorer la lisibilité des axes
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=9)
    
    # Format des ticks pour plus de clarté
    from matplotlib.ticker import ScalarFormatter
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    return fig, ax, ax2

def create_pdf_report(calculated_point, pipe_type, pipe_length, total_pressure_loss, C_value):
    """Crée un rapport PDF avec les résultats et le diagramme"""
    pdf = FPDF()
    pdf.add_page()
    
    # Titre
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Rapport de Calcul Hydraulique - Diagramme de Viveleau", 0, 1, 'C')
    pdf.ln(10)
    
    # Informations générales
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Paramètres du calcul:", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Type de canalisation: {pipe_type} (C={C_value})", 0, 1)
    pdf.cell(0, 8, f"Longueur de canalisation: {pipe_length} m", 0, 1)
    pdf.ln(5)
    
    # Résultats
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Résultats calculés:", 0, 1)
    pdf.set_font("Arial", '', 11)
    
    pdf.cell(0, 8, f"Débit: {calculated_point['Q_ls']:.1f} l/s ({calculated_point['Q_m3h']:.1f} m³/h)", 0, 1)
    pdf.cell(0, 8, f"Diamètre: DN {calculated_point['D']:.0f} mm", 0, 1)
    pdf.cell(0, 8, f"Vitesse: {calculated_point['V']:.2f} m/s", 0, 1)
    pdf.cell(0, 8, f"Perte de charge linéique: {calculated_point['J']:.4f} mCE/m", 0, 1)
    pdf.cell(0, 8, f"Perte de charge totale: {total_pressure_loss:.2f} mCE", 0, 1)
    
    pdf.ln(10)
    
    # Générer et ajouter le diagramme
    try:
        # Créer le diagramme
        fig, ax, ax2 = create_dynamic_viveleau_diagram(
            Q_point=calculated_point['Q_ls'], 
            J_point=calculated_point['J'],
            D_point=calculated_point['D'],
            V_point=calculated_point['V'],
            pipe_type=pipe_type
        )
        
        # Sauvegarder le diagramme dans un buffer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Fermer la figure pour libérer la mémoire
        plt.close(fig)
        
        # Ajouter le diagramme au PDF
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Diagramme de Viveleau:", 0, 1)
        pdf.ln(5)
        
        # Sauvegarder l'image temporairement pour FPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(buf.getvalue())
            image_path = tmp_img.name
        
        # Ajouter l'image au PDF (redimensionnée pour s'adapter à la page)
        pdf.image(image_path, x=10, y=pdf.get_y(), w=190)
        
        # Nettoyer le fichier temporaire
        os.unlink(image_path)
        
    except Exception as e:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, f"Erreur lors de la génération du diagramme: {str(e)}", 0, 1)
    
    # Ajouter le filigrane
    pdf.set_font("Arial", 'I', 40)
    pdf.set_text_color(200, 200, 200)  # Gris clair
    pdf.rotate(45)  # Rotation à 45 degrés
    pdf.text(60, 150, "By Viveleau 2025")
    pdf.rotate(0)  # Remettre à 0 degrés
    pdf.set_text_color(0, 0, 0)  # Remettre en noir
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, "Généré automatiquement par l'application Diagramme de Viveleau", 0, 1)
    
    return pdf

def main():
    st.set_page_config(page_title="Diagramme de Viveleau", layout="wide")
    
    st.title("📊 Diagramme de Viveleau Interactif")
    st.markdown("""
    Cet outil permet de déterminer les paramètres hydrauliques d'une conduite selon le diagramme de Viveleau.
    **Remplissez exactement 2 paramètres** pour calculer les 2 autres.
    """)
    
    # Initialisation de l'état de session pour l'unité par défaut
    if 'flow_unit' not in st.session_state:
        st.session_state.flow_unit = "m³/h"  # Unité par défaut
    
    # Création des colonnes pour l'interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Paramètres d'entrée")
        
        # Sélection du type de canalisation
        pipe_type = st.selectbox(
            "Type de canalisation:",
            options=list(HAZEN_WILLIAMS_COEFFICIENTS.keys()),
            index=0,  # Acier par défaut
            help="Coefficient C de Hazen-Williams selon le matériau"
        )
        
        # Afficher le coefficient C
        C_value = HAZEN_WILLIAMS_COEFFICIENTS[pipe_type]
        st.info(f"🔧 Coefficient C de Hazen-Williams: **{C_value}**")
        
        # Longueur de canalisation pour calcul des pertes totales
        pipe_length = st.number_input(
            "Longueur de canalisation (m)", 
            min_value=1.0, 
            max_value=10000.0, 
            value=100.0,
            help="Longueur totale pour calculer la perte de charge totale"
        )
        
        # Sélection de l'unité pour le débit - m³/h sélectionné par défaut
        flow_unit = st.radio("Unité pour le débit:", ["m³/h", "l/s"], 
                           index=0,  # m³/h en première position et sélectionné
                           horizontal=True,
                           key="flow_unit_selector")
        
        # Champs de saisie avec m³/h par défaut
        if flow_unit == "m³/h":
            Q_input = st.number_input("Débit (m³/h)", 
                                    min_value=0.36, 
                                    max_value=3600.0, 
                                    value=None, 
                                    placeholder="Ex: 36.0")
            Q_m3h_input = Q_input
        else:  # l/s
            Q_input = st.number_input("Débit (l/s)", 
                                    min_value=0.1, 
                                    max_value=1000.0, 
                                    value=None, 
                                    placeholder="Ex: 10.0")
            Q_m3h_input = Q_input * 3.6 if Q_input is not None else None
        
        D_input = st.number_input("Diamètre (mm)", 
                                min_value=10, 
                                max_value=2000, 
                                value=None, 
                                placeholder="Ex: 150")
        
        J_input = st.number_input("Perte de charge (mCE/m)", 
                                min_value=0.0001, 
                                max_value=0.1, 
                                value=None, 
                                format="%.4f", 
                                placeholder="Ex: 0.005")
        
        V_input = st.number_input("Vitesse (m/s)", 
                                min_value=0.1, 
                                max_value=5.0, 
                                value=None, 
                                placeholder="Ex: 1.5")
        
        # Affichage de la conversion rapide
        if Q_input is not None:
            if flow_unit == "l/s":
                st.info(f"💡 Conversion: {Q_input:.1f} l/s = {Q_input*3.6:.1f} m³/h")
            else:
                st.info(f"💡 Conversion: {Q_input:.1f} m³/h = {Q_input/3.6:.1f} l/s")
        
        # Vérification du nombre de paramètres saisis
        params = [Q_input, D_input, J_input, V_input]
        non_none_params = [p for p in params if p is not None]
        
        if len(non_none_params) == 2:
            try:
                # Calcul des paramètres manquants
                Q_ls_calc, Q_m3h_calc, D_calc, J_calc, V_calc, original_params, C_used = calculate_hazen_williams(
                    Q_input, D_input, J_input, V_input, flow_unit, pipe_type
                )
                
                # Vérifier si le calcul a réussi
                if Q_ls_calc is not None:
                    st.success("Calcul réussi !")
                    
                    # Calcul de la perte de charge totale
                    total_pressure_loss = J_calc * pipe_length
                    
                    # Déterminer quels paramètres ont été saisis et lesquels sont calculés
                    is_Q_input = original_params['Q'] is not None
                    is_D_input = original_params['D'] is not None
                    is_J_input = original_params['J'] is not None
                    is_V_input = original_params['V'] is not None
                    
                    # Affichage des résultats
                    st.header("Résultats calculés")
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        # Débit - AFFICHAGE avec m³/h en premier
                        if is_Q_input:
                            # Débit saisi par l'utilisateur
                            if flow_unit == "m³/h":
                                st.metric("Débit", f"{Q_input:.1f} m³/h", delta="saisi")
                                st.metric("Débit", f"{Q_ls_calc:.1f} l/s", delta="calculé")
                            else:
                                st.metric("Débit", f"{Q_input:.1f} l/s", delta="saisi")
                                st.metric("Débit", f"{Q_m3h_calc:.1f} m³/h", delta="calculé")
                        else:
                            # Débit calculé - m³/h affiché en premier
                            st.metric("Débit", f"{Q_m3h_calc:.1f} m³/h", delta="calculé")
                            st.metric("Débit", f"{Q_ls_calc:.1f} l/s", delta="calculé")
                        
                        # Diamètre
                        if is_D_input:
                            st.metric("Diamètre", f"{D_input:.0f} mm", delta="saisi")
                        else:
                            st.metric("Diamètre", f"{D_calc:.0f} mm", delta="calculé")
                    
                    with result_col2:
                        # Perte de charge
                        if is_J_input:
                            st.metric("Perte de charge", f"{J_input:.4f} mCE/m", delta="saisi")
                        else:
                            st.metric("Perte de charge", f"{J_calc:.4f} mCE/m", delta="calculé")
                        
                        # Vitesse
                        if is_V_input:
                            st.metric("Vitesse", f"{V_input:.2f} m/s", delta="saisi")
                        else:
                            st.metric("Vitesse", f"{V_calc:.2f} m/s", delta="calculé")
                    
                    # Affichage de la perte de charge totale
                    st.subheader("📏 Pertes de charge totales")
                    st.metric(
                        "Perte de charge totale", 
                        f"{total_pressure_loss:.2f} mCE", 
                        help=f"Perte de charge sur {pipe_length} m de canalisation"
                    )
                    
                    # Stockage des valeurs pour le graphique
                    st.session_state.calculated_point = {
                        'Q_ls': Q_ls_calc,
                        'Q_m3h': Q_m3h_calc,
                        'D': D_calc,
                        'J': J_calc,
                        'V': V_calc,
                        'pipe_type': pipe_type
                    }
                    st.session_state.pipe_length = pipe_length
                    st.session_state.total_pressure_loss = total_pressure_loss
                    st.session_state.C_value = C_used
                    
                    # Bouton pour exporter en PDF
                    st.subheader("📄 Export PDF")
                    pdf = create_pdf_report(
                        st.session_state.calculated_point, 
                        pipe_type, 
                        pipe_length, 
                        total_pressure_loss, 
                        C_used
                    )
                    
                    # Sauvegarder le PDF dans un fichier temporaire
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        pdf.output(tmp_file.name)
                        with open(tmp_file.name, "rb") as f:
                            pdf_data = f.read()
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="📥 Enregistrer PDF",
                        data=pdf_data,
                        file_name=f"calcul_hydraulique_viveleau_{pipe_length}m.pdf",
                        mime="application/pdf",
                        help="Télécharger le rapport complet au format PDF"
                    )
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_file.name)
                    
                else:
                    st.error("❌ Impossible de calculer les paramètres avec les valeurs fournies")
                    if 'calculated_point' in st.session_state:
                        del st.session_state.calculated_point
                
            except Exception as e:
                st.error(f"❌ Erreur dans le calcul: {str(e)}")
                if 'calculated_point' in st.session_state:
                    del st.session_state.calculated_point
                    
        elif len(non_none_params) > 2:
            st.warning("⚠️ Veuillez saisir exactement 2 paramètres")
            if 'calculated_point' in st.session_state:
                del st.session_state.calculated_point
        else:
            st.info("ℹ️ Veuillez saisir 2 paramètres pour effectuer le calcul")
            if 'calculated_point' in st.session_state:
                del st.session_state.calculated_point
    
    with col2:
        st.header("Diagramme de Viveleau")
        
        # Création du diagramme dynamique
        if 'calculated_point' in st.session_state:
            point = st.session_state.calculated_point
            fig, ax, ax2 = create_dynamic_viveleau_diagram(
                Q_point=point['Q_ls'], 
                J_point=point['J'],
                D_point=point['D'],
                V_point=point['V'],
                pipe_type=point.get('pipe_type', 'Acier')
            )
        else:
            # Diagramme par défaut sans point
            fig, ax, ax2 = create_dynamic_viveleau_diagram(pipe_type=pipe_type)
        
        # Affichage du graphique
        st.pyplot(fig)
        
        # Explications
        with st.expander("ℹ️ Comment utiliser le diagramme"):
            st.markdown("""
            **Instructions d'utilisation:**
            1. Choisissez le type de canalisation (coefficient C)
            2. Entrez la longueur de canalisation pour calculer les pertes totales
            3. Choisissez l'unité pour le débit (m³/h par défaut)
            4. Remplissez **exactement 2 paramètres** dans les champs de gauche
            5. Les 2 paramètres manquants seront automatiquement calculés
            6. Le point de fonctionnement apparaît en rouge sur le diagramme
            
            **Combinaisons possibles:**
            - Débit + Diamètre → Vitesse + Perte de charge
            - Débit + Perte de charge → Diamètre + Vitesse  
            - Diamètre + Perte de charge → Débit + Vitesse
            - Vitesse + Diamètre → Débit + Perte de charge
            - Vitesse + Perte de charge → Débit + Diamètre
            - **Débit + Vitesse → Diamètre + Perte de charge**
            
            **Échelles du diagramme:**
            - Abscisse (bas): Débit en litres par seconde (l/s) - échelle logarithmique
            - Abscisse (haut): Débit en mètres cubes par heure (m³/h) - échelle logarithmique  
            - Ordonnée: Perte de charge en mètres de colonne d'eau par mètre (mCE/m) - échelle logarithmique
            
            **Légende:**
            - Lignes bleues: Diamètres nominaux (DN)
            - Lignes pointillées rouges: Vitesses d'écoulement constantes (m/s)
            - Lignes vertes pointillées: Repères du point de fonctionnement
            - Point rouge: Point de fonctionnement calculé
            
            **Conversion:** 1 l/s = 3.6 m³/h
            """)
    
    # Pied de page
    st.markdown("---")
    st.caption(f"Basé sur la formule de Hazen-Williams • Type de canalisation: {pipe_type} (C={C_value}) • Débit par défaut en m³/h • 1 l/s = 3.6 m³/h • Diagramme de Viveleau")

if __name__ == "__main__":
    main()
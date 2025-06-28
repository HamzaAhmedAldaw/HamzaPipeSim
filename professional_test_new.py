#!/usr/bin/env python3
"""
Enhanced professional oil field dashboard with polished visualization
"""

import pipeline_sim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, FancyArrowPatch, Wedge
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib import patheffects
import matplotlib.patheffects as path_effects
from datetime import datetime
import seaborn as sns

# Set style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_simulation_data():
    """Create realistic simulation data"""
    
    # Well data
    wells = {
        'WELL-A1': {'x': -3.0, 'y': 2.0, 'pressure': 320, 'group': 'A', 'flow': 445},
        'WELL-A2': {'x': -3.0, 'y': 0.0, 'pressure': 310, 'group': 'A', 'flow': 425},
        'WELL-A3': {'x': -3.0, 'y': -2.0, 'pressure': 300, 'group': 'A', 'flow': 405},
        'WELL-B1': {'x': -1.0, 'y': 2.5, 'pressure': 330, 'group': 'B', 'flow': 465},
        'WELL-B2': {'x': -1.0, 'y': 0.5, 'pressure': 325, 'group': 'B', 'flow': 455},
        'WELL-B3': {'x': -1.0, 'y': -1.5, 'pressure': 315, 'group': 'B', 'flow': 435},
    }
    
    # Facility data
    facilities = {
        'MANIFOLD-A': {'x': -2.0, 'y': 0.0, 'pressure': 250, 'type': 'manifold'},
        'MANIFOLD-B': {'x': 0.0, 'y': 0.5, 'pressure': 260, 'type': 'manifold'},
        'PLATFORM': {'x': 2.0, 'y': 0.0, 'pressure': 180, 'type': 'platform'},
        'SEPARATOR': {'x': 3.0, 'y': 0.0, 'pressure': 150, 'type': 'separator'},
        'EXPORT': {'x': 4.0, 'y': 0.0, 'pressure': 120, 'type': 'export'},
    }
    
    # Combine all nodes
    all_nodes = {**wells, **facilities}
    
    # Calculate total flows
    total_flow_a = sum(w['flow'] for w in wells.values() if w['group'] == 'A')
    total_flow_b = sum(w['flow'] for w in wells.values() if w['group'] == 'B')
    total_flow = total_flow_a + total_flow_b
    
    # Pipe flows
    pipe_flows = {
        'MANIFOLD-A_to_PLATFORM': total_flow_a,
        'MANIFOLD-B_to_PLATFORM': total_flow_b,
        'PLATFORM_to_SEPARATOR': total_flow,
        'SEPARATOR_to_EXPORT': total_flow,
    }
    
    # Fluid properties
    fluid = {
        'oil_fraction': 0.75,
        'water_fraction': 0.20,
        'gas_fraction': 0.05,
        'density': 790.8,
        'viscosity': 0.82
    }
    
    return wells, facilities, all_nodes, pipe_flows, fluid

def create_professional_dashboard():
    """Create a truly professional dashboard"""
    
    # Get data
    wells, facilities, all_nodes, pipe_flows, fluid = create_simulation_data()
    
    # Create figure with dark background for professional look
    fig = plt.figure(figsize=(24, 14), facecolor='#1e1e1e')
    
    # Create grid with better proportions
    gs = fig.add_gridspec(3, 4, 
                         height_ratios=[2.5, 1, 1],
                         width_ratios=[1.5, 1.5, 1, 1],
                         hspace=0.25, wspace=0.2,
                         left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # 1. Main field schematic (spanning 2 columns)
    ax_field = fig.add_subplot(gs[0, :2], facecolor='#2e2e2e')
    draw_field_schematic(ax_field, wells, facilities, pipe_flows)
    
    # 2. 3D pressure surface
    ax_3d = fig.add_subplot(gs[0, 2:], projection='3d', facecolor='#2e2e2e')
    draw_3d_pressure_surface(ax_3d, all_nodes)
    
    # 3. Production trends
    ax_trends = fig.add_subplot(gs[1, :2], facecolor='#2e2e2e')
    draw_production_trends(ax_trends, wells)
    
    # 4. Well performance matrix
    ax_matrix = fig.add_subplot(gs[1, 2], facecolor='#2e2e2e')
    draw_performance_matrix(ax_matrix, wells)
    
    # 5. KPI gauges
    ax_kpi = fig.add_subplot(gs[1, 3], facecolor='#2e2e2e')
    draw_kpi_gauges(ax_kpi, wells, fluid)
    
    # 6. System diagram
    ax_system = fig.add_subplot(gs[2, 0], facecolor='#2e2e2e')
    draw_system_diagram(ax_system, all_nodes)
    
    # 7. Production pie chart
    ax_pie = fig.add_subplot(gs[2, 1], facecolor='#2e2e2e')
    draw_enhanced_pie_chart(ax_pie, wells)
    
    # 8. Data table
    ax_table = fig.add_subplot(gs[2, 2:], facecolor='#2e2e2e')
    draw_data_table(ax_table, wells, facilities, fluid)
    
    # Main title
    fig.text(0.5, 0.98, 'OFFSHORE FIELD PRODUCTION DASHBOARD', 
             ha='center', va='top', fontsize=28, weight='bold', color='white',
             fontfamily='Arial Black')
    
    # Subtitle with date
    fig.text(0.5, 0.955, f'Real-Time Monitoring System | {datetime.now().strftime("%B %d, %Y %H:%M UTC")}',
             ha='center', va='top', fontsize=14, color='#cccccc',
             fontfamily='Arial')
    
    # Company logo area (placeholder)
    fig.text(0.02, 0.98, 'PIPELINE-SIM™', ha='left', va='top', 
             fontsize=16, weight='bold', color='#00a8ff',
             fontfamily='Arial Black')
    
    return fig

def draw_field_schematic(ax, wells, facilities, pipe_flows):
    """Draw professional field schematic"""
    
    ax.set_xlim(-4, 5)
    ax.set_ylim(-3, 3.5)
    ax.set_aspect('equal')
    
    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Title
    ax.text(0.5, 1.05, 'Field Layout Schematic', transform=ax.transAxes,
            ha='center', fontsize=18, weight='bold', color='white')
    
    # Draw seabed
    seabed_y = -2.5
    ax.fill_between([-4, 5], [seabed_y-0.5, seabed_y-0.5], [-3, -3], 
                    color='#3d3d3d', alpha=0.7)
    ax.text(0, seabed_y-0.3, 'SEABED', ha='center', va='center',
            fontsize=10, color='#888888', weight='bold')
    
    # Draw pipelines with gradient effect
    connections = [
        ('WELL-A1', 'MANIFOLD-A', 'flowline'),
        ('WELL-A2', 'MANIFOLD-A', 'flowline'),
        ('WELL-A3', 'MANIFOLD-A', 'flowline'),
        ('WELL-B1', 'MANIFOLD-B', 'flowline'),
        ('WELL-B2', 'MANIFOLD-B', 'flowline'),
        ('WELL-B3', 'MANIFOLD-B', 'flowline'),
        ('MANIFOLD-A', 'PLATFORM', 'trunk'),
        ('MANIFOLD-B', 'PLATFORM', 'trunk'),
        ('PLATFORM', 'SEPARATOR', 'process'),
        ('SEPARATOR', 'EXPORT', 'export'),
    ]
    
    # Combine node data
    all_nodes = {**wells, **facilities}
    
    for from_node, to_node, pipe_type in connections:
        x1, y1 = all_nodes[from_node]['x'], all_nodes[from_node]['y']
        x2, y2 = all_nodes[to_node]['x'], all_nodes[to_node]['y']
        
        # Determine pipe style
        if pipe_type == 'flowline':
            color = '#00a8ff'
            width = 3
            alpha = 0.8
        elif pipe_type == 'trunk':
            color = '#0066cc'
            width = 5
            alpha = 0.9
        else:
            color = '#666666'
            width = 4
            alpha = 0.7
        
        # Draw pipe with shadow effect
        shadow = ax.plot([x1, x2], [y1, y2], '-', color='black', 
                        linewidth=width+2, alpha=0.3, zorder=1)[0]
        shadow.set_path_effects([patheffects.SimpleLineShadow(offset=(2, -2)),
                                patheffects.Normal()])
        
        ax.plot([x1, x2], [y1, y2], '-', color=color, 
               linewidth=width, alpha=alpha, zorder=2)
        
        # Add flow for major pipes
        if pipe_type in ['trunk', 'process', 'export']:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            pipe_name = f"{from_node}_to_{to_node}"
            if pipe_name in pipe_flows:
                flow = pipe_flows[pipe_name]
                ax.annotate(f'{flow:.0f} m³/h', (mid_x, mid_y),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', fontsize=10, color='white',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.8),
                           weight='bold')
    
    # Draw nodes
    for name, data in wells.items():
        x, y = data['x'], data['y']
        
        # Well symbol - professional oil derrick style
        well_color = '#00ff00' if data['group'] == 'A' else '#00ffff'
        
        # Base
        base = Circle((x, y), 0.2, facecolor=well_color, edgecolor='white',
                     linewidth=2, zorder=5)
        ax.add_patch(base)
        
        # Derrick structure
        ax.plot([x, x], [y, y+0.3], 'white', linewidth=3, zorder=6)
        ax.plot([x-0.1, x+0.1], [y+0.3, y+0.3], 'white', linewidth=3, zorder=6)
        
        # Label
        ax.text(x, y-0.4, name, ha='center', va='top', fontsize=11,
               color='white', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=well_color, 
                        alpha=0.7, edgecolor='white'))
        
        # Pressure
        ax.text(x, y-0.7, f'{data["pressure"]} bar', ha='center', va='top',
               fontsize=9, color='#cccccc')
    
    # Draw facilities
    for name, data in facilities.items():
        x, y = data['x'], data['y']
        
        if data['type'] == 'manifold':
            # Manifold - subsea template
            rect = FancyBboxPatch((x-0.3, y-0.2), 0.6, 0.4,
                                 boxstyle="round,pad=0.02",
                                 facecolor='#ffcc00', edgecolor='white',
                                 linewidth=2, zorder=5)
            ax.add_patch(rect)
            ax.text(x, y, 'M', ha='center', va='center',
                   fontsize=16, weight='bold', color='black')
            
        elif data['type'] == 'platform':
            # FPSO platform
            rect = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#888888', edgecolor='white',
                                 linewidth=3, zorder=5)
            ax.add_patch(rect)
            ax.text(x, y, 'FPSO', ha='center', va='center',
                   fontsize=12, weight='bold', color='white')
            
        elif data['type'] == 'separator':
            # Separator vessel
            circle = Circle((x, y), 0.3, facecolor='#4488ff', 
                          edgecolor='white', linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, 'SEP', ha='center', va='center',
                   fontsize=12, weight='bold', color='white')
            
        elif data['type'] == 'export':
            # Export terminal
            rect = Rectangle((x-0.3, y-0.3), 0.6, 0.6,
                           facecolor='#ff4444', edgecolor='white',
                           linewidth=2, zorder=5)
            ax.add_patch(rect)
            ax.text(x, y, 'EXP', ha='center', va='center',
                   fontsize=12, weight='bold', color='white')
        
        # Label
        ax.text(x, y+0.5, name, ha='center', va='bottom',
               fontsize=10, color='white', weight='bold')
    
    # Add scale
    ax.plot([3, 4], [-2.8, -2.8], 'white', linewidth=2)
    ax.text(3.5, -2.95, '1 km', ha='center', fontsize=10, color='white')
    
    # Add north arrow
    ax.annotate('', xy=(4.5, 2.5), xytext=(4.5, 2),
               arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax.text(4.5, 2.7, 'N', ha='center', fontsize=12, color='white', weight='bold')

def draw_3d_pressure_surface(ax, all_nodes):
    """Draw 3D pressure surface"""
    
    # Create grid
    x_range = np.linspace(-4, 5, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Interpolate pressure field
    Z = np.zeros_like(X)
    for name, data in all_nodes.items():
        x, y = data['x'], data['y']
        p = data['pressure']
        
        # Gaussian influence
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        influence = np.exp(-dist**2 / 2)
        Z += p * influence
    
    # Normalize
    Z = Z / (len(all_nodes) * 0.3)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8,
                          linewidth=0, antialiased=True)
    
    # Add contours
    ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=1)
    
    # Styling
    ax.set_xlabel('East-West (km)', color='white', fontsize=10)
    ax.set_ylabel('North-South (km)', color='white', fontsize=10)
    ax.set_zlabel('Pressure (bar)', color='white', fontsize=10)
    ax.set_title('Pressure Distribution Field', fontsize=16, color='white', 
                weight='bold', pad=20)
    
    # Set dark background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.2)
    
    # Set tick colors
    ax.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label('Pressure (bar)', color='white')
    cbar.ax.tick_params(colors='white')

def draw_production_trends(ax, wells):
    """Draw production trends with professional styling"""
    
    # Generate time series data
    hours = np.arange(0, 24, 0.5)
    
    # Base production profiles with realistic variations
    for name, data in wells.items():
        base_flow = data['flow']
        
        # Add realistic variations
        noise = np.random.normal(0, base_flow * 0.02, len(hours))
        trend = base_flow * (1 - 0.01 * hours / 24)  # Slight decline
        seasonal = base_flow * 0.05 * np.sin(2 * np.pi * hours / 24)
        
        production = trend + seasonal + noise
        
        # Smooth the curve
        from scipy.ndimage import gaussian_filter1d
        production = gaussian_filter1d(production, sigma=1.5)
        
        # Plot with gradient
        color = '#00ff00' if data['group'] == 'A' else '#00ffff'
        ax.plot(hours, production, linewidth=2, alpha=0.9, label=name, color=color)
        ax.fill_between(hours, production, alpha=0.2, color=color)
    
    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12, color='white')
    ax.set_ylabel('Production Rate (m³/h)', fontsize=12, color='white')
    ax.set_title('24-Hour Production Trends', fontsize=16, color='white', weight='bold')
    ax.grid(True, alpha=0.3, color='white')
    ax.set_facecolor('#2e2e2e')
    
    # Legend
    ax.legend(loc='upper right', frameon=True, facecolor='#3e3e3e', 
             edgecolor='white', fontsize=10)
    
    # Set tick colors
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_performance_matrix(ax, wells):
    """Draw well performance matrix"""
    
    # Create performance data
    well_names = list(wells.keys())
    metrics = ['Flow Rate', 'Pressure', 'Efficiency', 'Uptime']
    
    # Generate normalized performance data
    data = []
    for well in well_names:
        flow_norm = wells[well]['flow'] / 500  # Normalize
        pressure_norm = wells[well]['pressure'] / 350
        efficiency = 0.85 + np.random.uniform(0, 0.1)
        uptime = 0.95 + np.random.uniform(0, 0.04)
        data.append([flow_norm, pressure_norm, efficiency, uptime])
    
    data = np.array(data).T
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(well_names)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(well_names, rotation=45, ha='right', color='white')
    ax.set_yticklabels(metrics, color='white')
    
    # Add values
    for i in range(len(metrics)):
        for j in range(len(well_names)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    # Title
    ax.set_title('Well Performance Matrix', fontsize=14, color='white', weight='bold')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_kpi_gauges(ax, wells, fluid):
    """Draw KPI gauges"""
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Key Performance Indicators', ha='center', 
           fontsize=14, color='white', weight='bold')
    
    # Calculate KPIs
    total_production = sum(w['flow'] for w in wells.values())
    efficiency = 0.965
    uptime = 0.987
    
    # KPI 1: Total Production
    draw_gauge(ax, 2.5, 6.5, total_production/3000, 'Production', 
              f'{total_production:.0f} m³/h', '#00ff00')
    
    # KPI 2: System Efficiency
    draw_gauge(ax, 7.5, 6.5, efficiency, 'Efficiency', 
              f'{efficiency*100:.1f}%', '#ffcc00')
    
    # KPI 3: Uptime
    draw_gauge(ax, 2.5, 2.5, uptime, 'Uptime', 
              f'{uptime*100:.1f}%', '#00ccff')
    
    # KPI 4: Water Cut
    water_cut = fluid['water_fraction']
    draw_gauge(ax, 7.5, 2.5, 1-water_cut, 'Oil Content', 
              f'{(1-water_cut)*100:.0f}%', '#ff6600')

def draw_gauge(ax, x, y, value, label, text, color):
    """Draw a single gauge"""
    
    # Outer circle
    circle = Circle((x, y), 1.2, facecolor='#3e3e3e', edgecolor=color, linewidth=3)
    ax.add_patch(circle)
    
    # Arc showing value
    theta1 = 180
    theta2 = 180 - value * 180
    arc = Wedge((x, y), 1.1, theta1, theta2, facecolor=color, alpha=0.8)
    ax.add_patch(arc)
    
    # Center text
    ax.text(x, y+0.1, text, ha='center', va='center', 
           fontsize=11, color='white', weight='bold')
    
    # Label
    ax.text(x, y-0.5, label, ha='center', va='center', 
           fontsize=10, color='white')

def draw_system_diagram(ax, all_nodes):
    """Draw simplified system flow diagram"""
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'System Flow Diagram', ha='center', 
           fontsize=14, color='white', weight='bold')
    
    # Draw simplified flow
    components = [
        (2, 7, 'Wells\n(6)', '#00ff00'),
        (4, 7, 'Manifolds\n(2)', '#ffcc00'),
        (6, 7, 'FPSO', '#888888'),
        (8, 7, 'Export', '#ff4444'),
    ]
    
    for i, (x, y, label, color) in enumerate(components):
        # Box
        rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.8, edgecolor='white')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=10, color='white', weight='bold')
        
        # Arrow to next
        if i < len(components) - 1:
            ax.annotate('', xy=(components[i+1][0]-0.8, y), 
                       xytext=(x+0.8, y),
                       arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Add flow values
    ax.text(5, 5.5, 'Total Flow', ha='center', fontsize=12, 
           color='white', weight='bold')
    ax.text(5, 4.8, '2,660 m³/h', ha='center', fontsize=16, 
           color='#00ff00', weight='bold')
    ax.text(5, 4.2, '63,840 m³/day', ha='center', fontsize=12, 
           color='#00ff00')

def draw_enhanced_pie_chart(ax, wells):
    """Draw enhanced production distribution pie chart"""
    
    # Calculate group totals
    group_a_total = sum(w['flow'] for w in wells.values() if w['group'] == 'A')
    group_b_total = sum(w['flow'] for w in wells.values() if w['group'] == 'B')
    
    # Data
    sizes = [group_a_total, group_b_total]
    labels = ['Group A', 'Group B']
    colors = ['#00ff00', '#00ffff']
    explode = (0.05, 0.05)
    
    # Create pie with enhanced styling
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                      autopct='%1.1f%%', startangle=90,
                                      explode=explode, shadow=True,
                                      textprops={'weight': 'bold', 'color': 'white'})
    
    # Enhance text
    for text in texts:
        text.set_color('white')
        text.set_fontsize(12)
    
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    # Center circle for donut effect
    centre_circle = Circle((0, 0), 0.70, fc='#2e2e2e')
    ax.add_artist(centre_circle)
    
    # Center text
    total = sum(sizes)
    ax.text(0, 0, f'{total:.0f}\nm³/h', ha='center', va='center',
           fontsize=16, color='white', weight='bold')
    
    # Title
    ax.set_title('Production by Group', fontsize=14, color='white', weight='bold')

def draw_data_table(ax, wells, facilities, fluid):
    """Draw summary data table"""
    
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Production Summary', ha='center', 
           transform=ax.transAxes, fontsize=14, color='white', weight='bold')
    
    # Calculate summary data
    total_production = sum(w['flow'] for w in wells.values())
    oil_production = total_production * fluid['oil_fraction']
    water_production = total_production * fluid['water_fraction']
    gas_production = total_production * fluid['gas_fraction']
    
    # Create table data
    table_data = [
        ['Metric', 'Value', 'Unit'],
        ['Total Liquid Production', f'{total_production:.0f}', 'm³/h'],
        ['Oil Production', f'{oil_production:.0f}', 'm³/h'],
        ['Water Production', f'{water_production:.0f}', 'm³/h'],
        ['Gas Production', f'{gas_production*1000:.0f}', 'sm³/h'],
        ['Water Cut', f'{fluid["water_fraction"]*100:.1f}', '%'],
        ['Active Wells', '6', 'wells'],
        ['System Pressure', '120-330', 'bar'],
        ['API Gravity', '32.5', '°API'],
    ]
    
    # Create table
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center',
                    colWidths=[0.4, 0.3, 0.3])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color cells
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4e4e4e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#3e3e3e')
                cell.set_text_props(color='white')
            cell.set_edgecolor('white')

def main():
    """Main execution"""
    
    print("Generating Enhanced Professional Dashboard...")
    
    # Create dashboard
    fig = create_professional_dashboard()
    
    # Save high-resolution image
    filename = f'professional_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1e1e1e')
    print(f"Dashboard saved: {filename}")
    
    # Show
    plt.show()

if __name__ == "__main__":
    # Check for scipy
    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError:
        print("Note: Install scipy for smoother curves: pip install scipy")
    
    main()
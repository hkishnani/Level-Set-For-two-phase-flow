import numpy as np
import pyvista as pv
import plotly.express as px

import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from urllib.request import urlopen
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from plotly.subplots import make_subplots

from LS_funcs import H
from LS_funcs import Smooth_D_Delta as DD
from momentum import kappa

# Plotly template setup
import plotly.io as pio
pio.templates.default = "presentation"
# pio.renderers.default = 'colab'
pio.renderers.default = 'svg'

'''
:param dx   : cell width in x-direction
:param dy   : cell height in y-direction
:return mesh: Structured grid mesh of pv object
'''

def VISUALIZE_MESH(dx: np.array, dy: np.array)\
    -> pv.StructuredGrid:

    x_0 = 0.0
    y_0 = 0.0
    # Assuming that bottom left corner is (0.0,0.0)

    x_vec = np.array([x_0], dtype=np.float64)
    y_vec = np.array([y_0], dtype=np.float64)

    x_vec = np.append(x_vec, np.cumsum(dx))
    y_vec = np.append(y_vec, np.cumsum(dy))

    xx, yy = np.meshgrid(x_vec, y_vec, indexing='xy')
    zz     = np.zeros_like(xx)

    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    foo = pv.PolyData(points)

    mesh = pv.StructuredGrid()
    mesh.points = foo.points
    mesh.dimensions = [x_vec.shape[0], y_vec.shape[0], 1]

    mesh.plot(show_edges = True, clim = [-1,1], cpos='xy')
    print("mesh is returned from VISUALIZE_MESH")

    return mesh


def VISUALIZE_STREAMLINES(u:np.array, v:np.array,
                          X_c: np.array, Y_c: np.array,
                          c: tuple, r: float):
    fig, axs = plt.subplots(1,1)
    axs.streamplot(x=X_c, y = Y_c, u=u, v = v, density=1.0)

    circle = Circle(c, r)
    p = PatchCollection([circle], alpha = 0.4)
    p.set_array([100])  # For color
    axs.add_collection(p)
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.show()


def VISUALIZE_FIELD(phi: np.array, X:np.array, Y:np.array, title:str):
    fig = px.imshow(phi, origin='lower', text_auto=True,
                    labels=dict(x = "X_c", y = "Y_c", color = "Magnitude"),
                    title=title)
    fig.show()
    
def VISUALIZE_STREAMLINE_PLOTLY(u: np.array, v: np.array, p: np.array,
                     X: np.array, Y: np.array,
                     c: tuple, r: float, t: float,
                     title: str = 'Quiver plot'
                     ):
    X_g, Y_g = np.meshgrid(X, Y, indexing='xy')
    
    slicein = 4
    X_g = X_g[::slicein]
    Y_g = Y_g[::slicein]
    
    u = u[::slicein]
    v = v[::slicein]
    
    fig = make_subplots(rows=1, cols=1, 
                        subplot_titles=("Pressure Contour"))
    
    # fig.add_trace(ff.create_quiver(x =X_g, y =Y_g, u = u, v = v, 
    #                                scaleratio=1.0).data[0],
    #               row = 1, col = 1)
    
    # fig.update_yaxes( scaleanchor = "x", scaleratio = 1)

    fig.add_trace(go.Contour(z = p, x = X, y = Y,
                             contours=
                             dict(coloring ='fill', showlabels = True,
                                  labelfont = dict(size = 12, color = 'white')
                                  )
                             ), row = 1, col = 1
                  )
    fig.update_yaxes( scaleanchor = "x", scaleratio = 1)
    
    fig.add_shape(type = "circle",
                  xref = "x", yref="y",
                  x0 = c[0]-r, x1 = c[0]+r,
                  y0 = c[1]-r, y1 = c[1]+r,
                  line_color = "LightSeaGreen", 
                  fillcolor = "red",
                  opacity = 0.2
                  )
    
    fig.update_layout(title = f"Field Plots @ {t}",width = 1600, height = 800)
    
    fig.update_yaxes( scaleanchor = "x", scaleratio = 1)
    
    fig.show(config={'modeBarButtonsToAdd':['drawline', 'drawopenpath',
                                            'drawclosedpath', 'drawcircle',
                                            'drawrect', 'eraseshape'
                                       ]})
#%% Visualizing the volume
def VISUALIZE_VOL(phi: np.array, X:np.array, Y:np.array, 
                  dx:float, title:str, n_pic:int, fig):
    
    data = 1-H(phi, dx)
    
    # Vol = (1-data).sum() * dx**2

    fig.add_traces(go.Contour(z = data, 
                              x = X, y = Y,
                              contours_coloring='lines',
                              line_width=2,
                              showlegend=False, showscale=False
                              )
                   )
    
    fig.update_layout(
        template = "presentation",
        scene = {"xaxis": {"nticks": 5},
                 "zaxis": {"nticks": 5},
                 "aspectratio": {"x": 1, 
                                 "y": 1*(Y.max()/X.max()),
                                 "z": 1
                                 }
                },
        autosize = False,
        width = 700,
        height = 700,
        xaxis = dict(tickmode = 'linear',
                     tick0 = 0.0,
                     dtick = 0.1
                    ),
        yaxis = dict(tickmode = 'linear',
                     tick0 = 0.0,
                     dtick = 0.1
                    )
                     )
    
    if n_pic != 0:
        fig.update_layout(plot_bgcolor ='rgba(0,0,0,0)')

    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=2, linecolor='black',
                     mirror = True, title = "X",
                     range = [X[0]-dx/2, X[-1] + dx/2]
                     )
    # range = [0.30, 0.70]    # FOR X RANGE
    
    fig.update_yaxes(showgrid=False, zeroline=False,
                     showline=True, linewidth=2, linecolor='black',
                     mirror = True, title = "Y",
                     range = [Y[0]-dx/2, Y[-1] + dx/2]
                     )

    # range = [0.55, 0.95]   # FOR Y RANGE
    
    fig.show(renderers = "svg")
    
    return fig
    
#%% Draw Level Set functions
def draw_level_set(phi:np.array, X_c:np.array, Y_c:np.array, V0:float,
                   title_tuple:tuple, bound = 0.001):

    X,Y = np.meshgrid(X_c, Y_c)
    dx = X_c[0,1] - X_c[0,0]
    # Make Subplots specification
    fig = make_subplots(rows=1, cols = 2,
                       specs = [ [{"type":"scene"},{"type":"xy"}] ],
                       print_grid=False,
                       subplot_titles= title_tuple, 
                       shared_yaxes=True, shared_xaxes = True
                       )

    # Create a surface plot with contour projections on Z.min()-plane
    fig.add_trace(go.Surface(contours = {"z": {"show" : True,
                                               "start": -0.001,
                                               "end"  : 0.001,
                                               "size" : 0.001}},
                             z=phi, x=X, y = Y,
                             coloraxis="coloraxis",
                             showlegend=False,
                             showscale=True),
                  row = 1, col = 1)


    fig.update_traces(contours_z = dict(show = True,
                                    usecolormap = False,
                                    highlightcolor = "limegreen",
                                    project_z =True)
                     )
    
    # Create contours for interface detection
    fig.add_trace(go.Contour(z = H(phi,1.5*dx), 
                             x = X_c[0,:], y = Y_c[:,0], 
                             coloraxis="coloraxis"),
                  row = 1, col = 2)
    
    fig.update_yaxes(scaleanchor='x',
                     scaleratio=1,
                     constrain="domain")
    
    Vol = (1-H(phi, eps = 1.5*dx)).sum() * dx**2
    fig.update_layout(
        title_text = f"% change in Vol = {abs(1-Vol/V0)*100}",
                      scene = {"xaxis": {"nticks": 5},
                               "zaxis": {"nticks": 5},
                               "aspectratio": {"x": 1, 
                                               "y": 1*(Y_c.max()/X_c.max()),
                                               "z": 1
                                               }
                              },
                      autosize = True
                     )

    fig.show()
#%% Investigation plots
def invest_traces(phi: np.array, p:np.array,
                  X_c: np.array, Y_c: np.array,
                  dx:np.array, dy:np.array,
                  sigma: float,
                  title: str, V0: float,
                  L_x: float, L_y: float):
    
    X,Y = np.meshgrid(X_c, Y_c)
    
    ST_x, ST_y = kappa(phi, dx, dy)
    S_x = sigma*ST_x
    S_y = sigma*ST_y
    x_cb, y_cb = (0.22, 0.62, 1.02),(0.53, 0.53, 0.53)
    '''
    fig 1: Volume plot with obstacle
    fig 2: pressure plot
    fig 3: Surface Tension plot
    '''
    title_tuple = ("Volume Plot", "Pressure Plot", "σ.κ(φ).δ(φ).ṉ")
    
    fig = make_subplots(rows = 1, cols = 3,
                        specs = [ [{"type":"xy"}, {"type":"xy"},{"type":"xy"}] ],
                        print_grid=False,
                        subplot_titles= title_tuple, 
                        shared_yaxes=False, shared_xaxes = False,
                        horizontal_spacing = 0.33
                        )
    
    # Volume plot with obstacle-----------------------------------
    fig.add_trace(go.Contour(z = H(phi,1.5*dx.max()), 
                             x = X_c[0,:], y = Y_c[:,0],
                             colorbar=dict(
                             x=x_cb[0], y = y_cb[0],
                             title="H(φ)"
                             )),
                  row = 1, col = 1)
    # Volume plot with obstacle-----------------------------------
    
    # Pressure plot-----------------------------------------------
    fig.add_trace(go.Heatmap(z = p, 
                             x = X_c[0,:], y = Y_c[:,0], 
                             coloraxis="coloraxis",
                             colorbar=dict(
                             x=x_cb[1], y = y_cb[1],
                             title="P (Pa)"
                             )
                             ),
                  row = 1, col = 2)
    # Pressure plot-----------------------------------------------
    
    # Surface Tension plot----------------------------------------
    data = np.sqrt((S_x**2 + S_y**2)/2)
    ran = data.max() - data.min()
    fig.add_trace(go.Contour(z = data, 
                             x = X_c[0,:], y = Y_c[:,0],
                             coloraxis="coloraxis",
                             contours=dict(
                                           start= data.min(),
                                           end  = data.max(),
                                           size = ran/10
                                           ),
                             colorbar=dict(
                             x=x_cb[2], y = y_cb[2],
                             title="N.m"
                             )),
                  row = 1, col = 3)
    # Surface Tension plot----------------------------------------
    
    Vol = (1-H(phi, eps = 1.5*dx.max())).sum() * dx.max()**2
    fig.update_layout(
        title_text = f"ΔVol = {abs(1-Vol/V0)*100}%<==>{title}",
                      scene = {"xaxis": {"nticks": 10},
                               "zaxis": {"nticks": 10},
                               "aspectratio": {"x": 1, 
                                               "y": (L_y/L_x),
                                               "z": 1
                                               }
                              },
                      autosize = True
                     )
    
    # Update xaxis properties
    fig.update_xaxes(title_text="X", range=[0, L_x], 
                     showgrid=False, row=1, col=1)
    
    fig.update_xaxes(title_text="X", range=[0, L_x], 
                     showgrid=False, row=1, col=2)
    
    fig.update_xaxes(title_text="X", range=[0, L_x], 
                     showgrid=False, row=1, col=3)

    # Update yaxis properties
    fig.update_yaxes(title_text=None, range=[0, L_y], showgrid=False,
                     scaleanchor='x',
                     scaleratio=1,
                     constrain="domain", row=1, col=1)

    fig.update_yaxes(title_text=None, range=[0, L_y], showgrid=False,
                     scaleanchor='x',
                     scaleratio=1,
                     constrain="domain", row=1, col=2)
    
    fig.update_yaxes(title_text=None, range=[0, L_y], showgrid = False,
                     scaleanchor='x',
                     scaleratio=1,
                     constrain="domain" , row=1, col=3)

    fig.update_coloraxes(colorbar_borderwidth=1)
    # fig.update_scenes(aspectmode='data')

    fig.show()

def show_quiver(phi:np.array, X_c: np.array, Y_c:np.array,
                u:np.array, v:np.array, L_x:float, L_y:float,
                title:str, V0:float):

    dx = X_c[0,1] - X_c[0,0]
    Vol = (1 - H(phi, eps=1.5 * dx)).sum() * dx ** 2

    X, Y = np.meshgrid(X_c, Y_c)

    skip = (slice(None, None, 8), slice(None, None, 8))

    f = ff.create_quiver(x=X[skip], y=Y[skip],
                         u=u[skip], v=v[skip],
                         arrow_scale=0.3,
                         scale=0.005,
                         line=dict(width=2)
                        )

    trace1 = f.data[0]

    trace2 = go.Contour(z = H(phi, eps=1.5*dx), x = X_c[0,:], y = Y_c[:,0],
                        contours=dict(showlabels = True,
                                      labelfont = dict(size = 15, color = 'black')
                                      ),
                        colorscale='redor'
                        )

    data = [trace1, trace2]

    fig = go.FigureWidget(data)

    fig.update_layout(
        title_text = f"ΔVol = {abs(1-Vol/V0)*100}% @ {title}",
        scene={"xaxis": {"nticks": 5},
               "zaxis": {"nticks": 5},
               "aspectratio": {"x": 1,
                               "y": (L_y / L_x),
                               "z": 1
                               }
               },
        autosize=False,
        width=600, height=1000
    )

    fig.update_xaxes(range = [0, L_x], showgrid = False)

    fig.update_yaxes(range = [0, L_y], showgrid = False,
                     scaleanchor='x',
                     scaleratio=1,
                     constrain="domain")
    fig.show()
    
def PLOT_VORT_VOL_N_PRESS(phi:np.array, X_c: np.array, Y_c:np.array,
                          u:np.array, v:np.array, L_x:float, L_y:float,
                          P: np.array,
                          title:str, V0:float):
    '''
    This subroutine provides 2 plots of half of domain's length only
    
    First:: Quiver + Volume (on left half) and Vorticity (on right half)
    
    Second:: Pressure with fixed scale
    '''
    # Initial Volume calculation
    dx = X_c[0,1] - X_c[0,0]
    Vol = (1 - H(phi, eps=1.5 * dx)).sum() * dx ** 2

    # Creating Full mesh
    X, Y = np.meshgrid(X_c, Y_c)
    
    # Step 1: Add trace on subplot for Quiver
    X_crop = X_c.size//2
    Y_crop = Y_c.size//2
    
    CS = ("Oryel", "RdBu")
    
    # First comes quiver
    skip_q = (slice(0, Y_crop, 8), slice(0, X_crop, 8))
    f = ff.create_quiver(x=X[skip_q], y=Y[skip_q],
                         u=u[skip_q], v=v[skip_q],
                         arrow_scale=0.2,
                         scale=0.007,
                         line=dict(width=1.5),
                         line_color = "black"
                        )
    trace1 = f.data[0]

    # Second comes Contour Plot on left
    trace2 = go.Contour(z = H(phi[0:Y_crop, 0:X_crop], eps=1.5*dx),
                        x = X_c[0,0:X_crop], y = Y_c[0:Y_crop,0],
                        contours=dict(showlabels = True,
                                      labelfont = dict(size = 20, color = 'black')
                                      ),
                        colorscale=CS[0], colorbar_x = 0.1,
                        showlegend=False, showscale=False
                        )

    # opacity=0.25, showlegend=False, showscale=False,
    
    # Third comes contour
    trace3 = go.Contour(z = P[0:Y_crop, X_crop:],
                        x = X_c[0,X_crop:], y = Y_c[0:Y_crop,0],
                        contours=dict(start = 0.0, end = 1.0, size = 0.05,
                            showlabels = True,
                            labelfont = dict(size = 20, color = 'black')
                                      ),
                        colorscale=CS[1], colorbar_x = 1.0
                        )
    
    data = [trace1, trace2, trace3]

    fig = go.FigureWidget(data)

    
    fig.update_layout(
        title_text = f"ΔVol = {round(abs(1-Vol/V0)*100,3)}% @ {title}",
        scene={"xaxis": {"nticks": 10},
               "yaxis": {"nticks": 10},
               "aspectratio": {"x": 1,
                               "y": (L_y / L_x),
                               "z": 1
                               }
               },
        autosize=False, width = 600, height = 1000
    )

    fig.update_xaxes(range = [0, L_x], showgrid = False)

    fig.update_yaxes(range = [0, Y_c[0:Y_crop,0][-1] + dx], showgrid = False,
                      scaleanchor='x',
                      scaleratio=1,
                      constrain="domain")
    
    # fig.update_layout(plot_bgcolor ='rgba(0,0,0,0)')
    
    fig.show(renderers = "svg")
    

# ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
#  'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
#  'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
#  'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
#  'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
#  'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
#  'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
#  'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
#  'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
#  'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
#  'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
#  'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
#  'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
#  'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
#  'ylorrd']




    
def INITIALIZE_DOMAIN(phi:np.array, X_c: np.array, Y_c:np.array,
                      u:np.array, v:np.array, L_x:float, L_y:float,
                      P: np.array,
                      title:str, V0:float):
    '''
    This subroutine provides 2 plots of half of domain's length only
    
    First:: Quiver + Volume (on left half) and Vorticity (on right half)
    
    Second:: Pressure with fixed scale
    '''
    # Initial Volume calculation
    dx = X_c[0,1] - X_c[0,0]
    Vol = (1 - H(phi, eps=1.5 * dx)).sum() * dx ** 2

    # Creating Full mesh
    X, Y = np.meshgrid(X_c, Y_c)
    
    # Step 1: Add trace on subplot for Quiver
    X_crop = X_c.size//2
    Y_crop = Y_c.size//2
    
    CS = ("Oryel", "RdBu")
    
    # First comes quiver
    skip_q = (slice(0, Y_crop, 8), slice(0, X_crop, 8))
    f = ff.create_quiver(x=X[skip_q], y=Y[skip_q],
                         u=u[skip_q], v=v[skip_q],
                         arrow_scale=0.2,
                         scale=0.007,
                         line=dict(width=1.5),
                         line_color = "black"
                        )
    trace1 = f.data[0]

    # Second comes Contour Plot on left
    trace2 = go.Contour(z = H(phi[0:Y_crop, 0:X_crop], eps=1.5*dx),
                        x = X_c[0,0:X_crop], y = Y_c[0:Y_crop,0],
                        contours_coloring='lines',
                        contours=dict(showlabels = False,
                                      labelfont = dict(size = 20, 
                                                       color = 'black')
                                      ),
                        colorscale=CS[0], colorbar_x = 0.1,
                        showlegend=False, showscale=False
                        )

    # opacity=0.25, showlegend=False, showscale=False,
    
    # Third comes contour
    trace3 = go.Contour(z = np.round(P[0:Y_crop, X_crop:], 5),
                        x = X_c[0,X_crop:], y = Y_c[0:Y_crop,0],
                        contours_coloring='lines',
                        contours=dict(start = 0.0, end = 1.0, size = 0.05,
                            showlabels = True,
                            labelfont = dict(size = 20, color = 'black')
                                      ),
                        colorscale=CS[1], colorbar_x = 1.0,
                        colorbar=dict(
                            title="<----Color scale for Pressure (pa)---->",
                            titleside="right",
                                      )
                        )
    
    data = [trace1, trace2, trace3]

    fig = go.FigureWidget(data)

    
    fig.update_layout(
        title_text = title,
        scene={"xaxis": {"nticks": 10},
               "yaxis": {"nticks": 10},
               "aspectratio": {"x": 1,
                               "y": (L_y / L_x),
                               "z": 1
                               }
               },
        autosize=False, width = 600, height = 1000
    )

    fig.update_xaxes(range = [0, L_x], showgrid=False, 
                     zeroline=False, showline=True, 
                     linewidth=2, linecolor='black',
                     mirror = True, title = "X [FULL]")

    fig.update_yaxes(range = [0, Y_c[0:Y_crop,0][-1] + dx], showgrid = False,
                      scaleanchor='x',
                      scaleratio=1,
                      constrain="domain",
                      zeroline=False, showline=True, 
                      linewidth=2, linecolor='black',
                      mirror = True, title = "Y [HALF]")
    
    fig.add_annotation(x=0.004, y=0.005,
            xshift=0, arrowsize=2,
            font=dict(color="black", size=25),
            textangle=-90,
            text="ρ = 1.0 kg/m³",
            showarrow=False,
            arrowhead=1
            )
    
    fig.add_annotation(x=0.004, y=0.015,
            xshift=0, arrowsize=2,
            font=dict(color="black", size=25),
            textangle=-90,
            text="ρ = 2.0 kg/m³",
            showarrow=False,
            arrowhead=1
            )
    
    fig.add_annotation(x=0.002, y=0.01,
            xshift=0, arrowsize=2,
            font=dict(color="blue", size=30),
            textangle=-90,
            text="VOLUME + QUIVER",
            showarrow=False,
            arrowhead=1,
            bordercolor="blue", 
            bgcolor='rgba(255,255,255,255)',
            borderpad=10
            )
    
    fig.add_annotation(x=0.008, y=0.01,
            xshift=0, arrowsize=2,
            font=dict(color="blue", size=30),
            textangle=-90,
            text="PRESSURE CONTOUR",
            showarrow=False,
            arrowhead=1,
            bordercolor="blue", 
            bgcolor='rgba(255,255,255,255)',
            borderpad=10
            )
    
    fig.add_annotation(x=0.005, y=0.02,
            xshift=0, arrowsize=2,
            font=dict(color="green", size=30),
            textangle=0,
            text='''BROKEN DOMAIN''',
            showarrow=False,
            arrowhead=1,
            bordercolor="green", 
            bgcolor='rgba(255,255,255,255)',
            borderpad=5
            )
    
    fig.add_vline(x = 0.005)
    # fig.update_layout(plot_bgcolor ='rgba(0,0,0,0)')

    
    fig.show(renderers = "svg")
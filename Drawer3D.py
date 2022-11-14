#####################################################################################
#
#
# Title: Drawer 3D
#
# Date: 21 October 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Richard Elvira, Jesus Bermudez, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


def drawPoints(fig:go.Figure,pointArray:np.array,marker:dict):
    """
	Draw points in 3D. 
           fig: Figure where the points are drawed
           pointArray: np.array Nx3 with the point cloud
           marker: marker style to draw.
    """                  
    x = np.transpose(pointArray[:,0])
    y = np.transpose(pointArray[:,1])
    z = np.transpose(pointArray[:,2])
    names = np.array(range(1, pointArray.shape[0]+1)).astype(str)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode='markers+text',
                                   marker=marker,
                                   text=names,
                                   textposition="bottom center"))


def drawCamera(fig:go.Figure,T_w_c:np.array,):            
    """
	Draw camera as cones in 3D. 
           fig: Figure where the points are drawed
           T_w_c: pose drawed.
    """                
    fig.add_trace(go.Cone(x=[T_w_c[0,3]], y=[T_w_c[1,3]], z=[T_w_c[2,3]], u=[-0.7*T_w_c[0,2]], v=[-0.7*T_w_c[1,2]], w=[-0.7*
    T_w_c[2,2]]))


def drawLine(fig:go.Figure,initialPoint:np.array,finalPoint:np.array,name:str,linedict:dict):
    """
	Draw line between two points in 3D. 
        fig: Figure where the points are drawed
        initialPoint: initial point. transposed vector 1x3
        finalPoint: final point drawed. transposed vector 1x3
	linedict: line style
    """         
    stack = np.vstack((initialPoint,finalPoint))
    x = np.transpose(stack[:,0])
    y = np.transpose(stack[:,1])
    z = np.transpose(stack[:,2])
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,
                                   mode='lines+text',
                                   line=linedict,
                                   text=[name, ""],
                                   textposition="bottom center"))


def drawRefSystem(fig:go.Figure,T_w_c:np.array, name:str):
    """
	Create a basis system with line in 3D. based in the pose T_w_c
        fig: Figure where the points are drawed
        T_w_c: pose drawed.
    """       
    zero = T_w_c[0:3,3]
    xaxis =  T_w_c[0:3,0]+zero
    yaxis =  T_w_c[0:3,1]+zero
    zaxis =  T_w_c[0:3,2]+zero

    drawLine(fig,zero,xaxis,name,linedict=dict(
                                      color='red',
                                      width=2
                                    ))
    drawLine(fig,zero,zaxis,"",linedict=dict(
                                      color='blue',
                                      width=2
                                    ))
    drawLine(fig,zero,yaxis,"",linedict=dict(
                                      color='green',
                                      width=2
                                    ))
    
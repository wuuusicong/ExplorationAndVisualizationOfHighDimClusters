import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import interp1d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import plotly.graph_objects as go

import alphashape
import shapely

import plotly.io as pio
pio.renderers.default = "png"

class PolygonsGenerator:
    POLY = 'poly'
    SMOOTH_POLY = 'smooth_poly'
    INNER_POINTS = 'inner_points'
    Z_DIST = 'z_dist'
    X_COL = 'X'
    Y_COL = 'Y'
    Z_COL = 'Z'
    POLY_COL = 'Poly'
    FEAT_COLS = [X_COL, Y_COL, Z_COL]

    DEFAULT_PLOTLY_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                             'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                             'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                             'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                             'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

    def __init__(self, smooth_iter=3, random_points=200):
        """
        :param smooth_iter:
        :param random_points:
        """
        self.smooth_iter = smooth_iter
        self.random_points = random_points

    def smooth_poly_Chaikins_corner_cutting_iter(self, poly):
        new_poly = poly[:]
        for i in range(self.smooth_iter):
            new_poly = PolygonsGenerator.smooth_poly_Chaikins_corner_cutting(new_poly, True)
        return new_poly

    @staticmethod
    def smooth_poly_Chaikins_corner_cutting(poly, append_first_point):
        """
        poly is list of lists
        example: poly1 = [
        [3,3],
        [4,4],
        [5,4],
        [5,7],
        [6,8],
        [7,5],
        [6,3],
        [5,2],
        [4,2],
        [3,3]
        ]
        Based on https://stackoverflow.com/questions/27642237/smoothing-a-2-d-figure
        Q(i) = (3/4)P(i) + (1/4)P(i+1)
        R(i) = (1/4)P(i) + (3/4)P(i+1)
        """
        new_poly = []
        for i in range(len(poly)-1):
            q_i = [0.75 * poly[i][0] + 0.25 * poly[i+1][0], 0.75 * poly[i][1] + 0.25 * poly[i+1][1]]
            r_i = [0.25 * poly[i][0] + 0.75 * poly[i+1][0], 0.25 * poly[i][1] + 0.75 * poly[i+1][1]]
            new_poly.extend([q_i, r_i])
        # append first point for smoothness
        if append_first_point:
            new_poly.append(new_poly[0])
        return new_poly

    @staticmethod
    def random_points_inside_polygon_and_3d(poly, number_of_points, loc, scale, num_outliers=0):
        """
        """
        # create shapely objects
        _poly = Polygon(poly)
        # get bounding box of polygon
        minx, miny, maxx, maxy = _poly.bounds
        # generate random points within the bounding box
        random_points = []
        while len(random_points) < number_of_points:
            x = np.random.uniform(low=minx, high=maxx)
            y = np.random.uniform(low=miny, high=maxy)
            z = np.random.normal(loc=loc, scale=scale)
            if _poly.contains(Point(x, y)):
                random_points.append([x, y, z])
        outliers=[]
        while len(outliers) < num_outliers:
            x = np.random.uniform(low=minx, high=maxx) + (maxx-minx)*3
            y = np.random.uniform(low=miny, high=maxy) + (maxy-miny)*3
            z = np.random.normal(loc=loc, scale=scale)
            outliers.append([x,y,z])
        return random_points + outliers

    def preprocess_polygons(self, polys, title='', num_outliers=0, make_plots=True):
        if isinstance(self.random_points, int):
            random_points = [self.random_points] * len(polys)
        else:
            random_points = self.random_points

        for i in range(len(polys)):
            polys[i][PolygonsGenerator.SMOOTH_POLY] = np.array(self.smooth_poly_Chaikins_corner_cutting_iter(
                polys[i][PolygonsGenerator.POLY]))
            polys[i][PolygonsGenerator.INNER_POINTS] = np.array(PolygonsGenerator.random_points_inside_polygon_and_3d(
                polys[i][PolygonsGenerator.SMOOTH_POLY],
                random_points[i],
                loc=polys[i][PolygonsGenerator.Z_DIST][0],
                scale=polys[i][PolygonsGenerator.Z_DIST][1],
                num_outliers=num_outliers))

        if make_plots:
            # Visualize 2D polygons
            #     color=iter(plt.cm.rainbow(np.linspace(0,1,len(_polys)*2)))
            color = iter(PolygonsGenerator.DEFAULT_PLOTLY_COLORS)
            fig = go.Figure()
            # Add traces
            for i in range(len(polys)):
                c = next(color)
                #         fig.add_trace(go.Scatter(x=_polys[i][SMOOTH_POLY][:,0],
                #                                  y=_polys[i][SMOOTH_POLY][:,1],
                #                                 mode='lines',
                #                                 name=f'poly{i}',
                # #                                 marker_color=f'rgba({c[0]}, {c[1]}, {c[2]}, {c[3]})'))
                #                                 marker_color=c))
                fig.add_trace(go.Scatter(x=polys[i][PolygonsGenerator.INNER_POINTS][:, 0],
                                         y=polys[i][PolygonsGenerator.INNER_POINTS][:, 1],
                                         mode='markers',
                                         name=f'poly{i}'))
            #                                 marker_color=f'rgba({c[0]}, {c[1]}, {c[2]}, {c[3]})'))
            #                                  marker_color=c))
            fig.update_layout(title=f'{title}: 2D Visualization')
            fig.show()

            # Visualize 3D
            #     color=iter(plt.cm.rainbow(np.linspace(0,1,len(_polys)*2)))
            color = iter(PolygonsGenerator.DEFAULT_PLOTLY_COLORS)
            fig = go.Figure()
            # Add traces
            for i in range(len(polys)):
                c = next(color)
                fig.add_trace(go.Scatter3d(x=polys[i][PolygonsGenerator.INNER_POINTS][:, 0],
                                           y=polys[i][PolygonsGenerator.INNER_POINTS][:, 1],
                                           z=polys[i][PolygonsGenerator.INNER_POINTS][:, 2],
                                           mode='markers',
                                           name=f'poly{i}'))
            #                                  marker_color=f'rgba({c[0]}, {c[1]}, {c[2]}, {c[3]})'))
            #                                    marker_color=c))
            fig.update_layout(title=f'{title}: 3D Visualization')
            fig.show()

        # Create df for dim-reduction
        dfs = []
        for i, poly in enumerate(polys):
            tmp_df = pd.DataFrame(poly[PolygonsGenerator.INNER_POINTS])
            feature_cols = [PolygonsGenerator.X_COL, PolygonsGenerator.Y_COL, PolygonsGenerator.Z_COL]
            tmp_df.columns = feature_cols
            tmp_df[PolygonsGenerator.POLY_COL] = i  #f' {i}'
            dfs.append(tmp_df)
        df = pd.concat(dfs)

        return df, feature_cols, PolygonsGenerator.POLY_COL

class PolygonsFactory:
    @staticmethod
    def get_polygons(polygons_name):
        pg = PolygonsGenerator()
        if polygons_name == 'fists_no_overlap':
            polys = [
                {PolygonsGenerator.POLY: [
                    [3, 3],
                    [3.5, 4],
                    [4, 4],
                    [5, 4],
                    [5, 7],
                    [6, 8],
                    [7, 5],
                    [6, 3],
                    [5, 2],
                    [4, 2],
                    [3.5, 2],
                    [3, 3],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 1)},
                {PolygonsGenerator.POLY: [
                    [0, 3],
                    [0, 5],
                    [1, 7],
                    [3, 9],
                    [5, 10],
                    [7, 10],
                    [6, 9],
                    [5, 9],
                    [4, 7],
                    [4, 5],
                    [3, 4],
                    [2, 3],
                    [3, 2],
                    [2, 1],
                    [0, 3],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 1)},
            ]
            return pg.preprocess_polygons(polys)
        elif polygons_name == 'cross':
            polys = [
                {PolygonsGenerator.POLY: [
                    [2, 0],
                    [3, 1],
                    [4, 2],
                    [5, 1],
                    [6, 0],
                    [5, -1],
                    [4, -2],
                    [3, -1],
                    [2, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (5, 1)},

                {PolygonsGenerator.POLY: [
                    [2, 0],
                    [3, 1],
                    [4, 2],
                    [5, 1],
                    [6, 0],
                    [5, -1],
                    [4, -2],
                    [3, -1],
                    [2, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},

                ####################
                {PolygonsGenerator.POLY: [
                    [12, 0],
                    [13, 1],
                    [14, 2],
                    [15, 1],
                    [16, 0],
                    [15, -1],
                    [14, -2],
                    [13, -1],
                    [12, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},
                ####################

                {PolygonsGenerator.POLY: [
                    [-2, 0],
                    [-3, 1],
                    [-4, 2],
                    [-5, 1],
                    [-6, 0],
                    [-5, -1],
                    [-4, -2],
                    [-3, -1],
                    [-2, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (5, 1)},

                {PolygonsGenerator.POLY: [
                    [-2, 0],
                    [-3, 1],
                    [-4, 2],
                    [-5, 1],
                    [-6, 0],
                    [-5, -1],
                    [-4, -2],
                    [-3, -1],
                    [-2, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},

                ######################
                {PolygonsGenerator.POLY: [
                    [-12, 0],
                    [-13, 1],
                    [-14, 2],
                    [-15, 1],
                    [-16, 0],
                    [-15, -1],
                    [-14, -2],
                    [-13, -1],
                    [-12, 0],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},
                ######################

                {PolygonsGenerator.POLY: [
                    [0, 5],
                    [-1, 7],
                    [-2, 9],
                    [-1, 11],
                    [0, 20],
                    [1, 11],
                    [2, 9],
                    [1, 7],
                    [0, 5],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},

                {PolygonsGenerator.POLY: [
                    [0, -5],
                    [-1, -7],
                    [-2, -9],
                    [-1, -11],
                    [0, -20],
                    [1, -11],
                    [2, -9],
                    [1, -7],
                    [0, -5],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None, PolygonsGenerator.Z_DIST: (-5, 1)},
            ]
            return pg.preprocess_polygons(polys)
        elif polygons_name == 'simple_overlap':
            polys = [
                {PolygonsGenerator.POLY: [
                    [1,2],
                    [3,4],
                    [6,4],
                    [8,2],
                    [6,0],
                    [3,0],
                    [1,2]
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 1)},
                {PolygonsGenerator.POLY: [
                    [1, 2],
                    [3, 4],
                    [5, 2],
                    [3, 0],
                    [1, 2]
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 3)},
            ]
            return pg.preprocess_polygons(polys)
        elif polygons_name == 'dense_in_sparse':
            polys = [
                {PolygonsGenerator.POLY: [
                    [0, 2],
                    [3, 6],
                    [7, 2],
                    [3, -2],
                    [0, 2]
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 1)},
                {PolygonsGenerator.POLY: [
                    [2.9, 2],
                    [3, 3.1],
                    [3.1, 2],
                    [3, 2.9],
                    [2.9, 2]
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 0.1)},
            ]
            return pg.preprocess_polygons(polys)
        elif polygons_name == 'hourglass':
            polys = [
                {PolygonsGenerator.POLY: [
                    [1,2],
                    [3,4],
                    [5,2.2],
                    [7,4],
                    [9,2],
                    [7,0],
                    [5, 1.8],
                    [3,0],
                    [1,2]
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 0.1)},
                {PolygonsGenerator.POLY: [
                    [4.8,2.5],
                    [5.2,2.5],
                    [5.2,1.5],
                    [4.8,1.5],
                    [4.8,2.5],
                ],
                    PolygonsGenerator.SMOOTH_POLY: None, PolygonsGenerator.INNER_POINTS: None,
                    PolygonsGenerator.Z_DIST: (0, 0.01)},
            ]
            return pg.preprocess_polygons(polys)
        else:
            raise Exception(f'Unsupported polygons {polygons_name}')


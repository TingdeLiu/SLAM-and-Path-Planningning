# Python routines to inspect a ikg LEGO robot logfile.
# Author: Claus Brenner, 28 OCT 2012
# 21 SEP 2020 CB  Modified for Python3.
# 05 OCT 2020 CB  Modified: this version is for jupyter notebooks and ipywidgets,
#                 it uses matplotlib instead of Tk drawing.
# (c) 05 OCT 2020 Claus Brenner
from lego_robot import *
from math import sin, cos, pi, ceil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import set_matplotlib_formats


class DrawableObject(object):
    def draw(self, at_step):
        print("To be overwritten - will draw a certain point in time:", at_step)

    def background_draw(self):
        print("Background draw.")


class Trajectory(DrawableObject):
    def __init__(self, points,
                 #canvas,
                 #world_extents, canvas_extents,
                 standard_deviations = [],
                 background_color = "gray", cursor_color = "red",
                 position_stddev_color = "green",
                 theta_stddev_color = "#ffc0c0"):
        self.points = np.array(points)
        self.standard_deviations = np.array(standard_deviations)
        self.background_color = background_color
        self.cursor_color = cursor_color
        self.position_stddev_color = position_stddev_color
        self.theta_stddev_color = theta_stddev_color

    def background_draw(self):
        if self.points.size > 0:
            # Plot line.
            plt.plot(self.points[:,0], self.points[:,1],
                     color=self.background_color, linewidth=0.5, zorder=10)
            # Plot dots at each position. s is maker size in points**2.
            plt.scatter(self.points[:,0], self.points[:,1],
                        color=self.background_color, s=4, zorder=10)

    def draw(self, at_step):
        if at_step < len(self.points):
            p = self.points[at_step]
            # Draw current position (point).
            plt.scatter(p[0], p[1], color=self.cursor_color, s=8, zorder=30)

            # Draw error ellipse for position.
            if at_step < len(self.standard_deviations):
                stddev = self.standard_deviations[at_step]
                ell = matplotlib.patches.Ellipse(
                    p[0:2], 2.0*stddev[1], 2.0*stddev[2], np.rad2deg(stddev[0]),
                    color=self.position_stddev_color, fill=False, linewidth=1,
                    zorder=20)
                plt.gca().add_patch(ell)

            if len(p) > 2:
                # Draw heading direction (a short line).
                l = matplotlib.lines.Line2D(
                    (p[0], p[0]+200.0*cos(p[2])), (p[1], p[1]+200.0*sin(p[2])),
                    color=self.cursor_color, linewidth=0.5, zorder=30)
                plt.gca().add_line(l)

                # Draw heading standard deviation.
                if at_step < len(self.standard_deviations) and\
                   len(self.standard_deviations[0]) > 3:
                    angle = np.rad2deg(
                        min(self.standard_deviations[at_step][3], pi))
                    heading = np.rad2deg(p[2])
                    wedge = matplotlib.patches.Wedge(p[0:2], 150.0,
                        heading-angle, heading+angle,
                        color=self.theta_stddev_color, zorder=20)
                    plt.gca().add_patch(wedge)


class ScannerData(DrawableObject):
    def __init__(self, list_of_scans):
        # Convert polar scanner measurements into xy form, store in
        # self.scan_polygons.
        # Note that x will be up and y will be left, so that we use
        # (-d*sin(a), d*cos(a)) instead of (d*cos(a), d*sin(a)).
        self.scan_polygons = []
        for s in list_of_scans:
            polar = ((LegoLogfile.beam_index_to_angle(i), d) \
                     for (i,d) in enumerate(s))
            cartesian = [ (-d*sin(a), d*cos(a)) for (a,d) in polar ]
            self.scan_polygons.append(np.array([(0,0)] + cartesian + [(0,0)]))

    def background_draw(self):
        pass

    def draw(self, at_step):
        if at_step < len(self.scan_polygons):
            poly = matplotlib.patches.Polygon(self.scan_polygons[at_step],
                                              color="blue", zorder=5)
            plt.gca().add_patch(poly)


class Landmarks(DrawableObject):
    def __init__(self, landmarks, color = "gray"):
        self.landmarks = landmarks
        self.color = color

    def background_draw(self):
        for e in self.landmarks:
            if e[0]=='C':
                plt.gca().add_patch(
                    matplotlib.patches.Circle(e[1:3], radius=e[3],
                                              color=self.color, zorder=0))

    def draw(self, at_step):
        # Landmarks are background only.
        pass


class Points(DrawableObject):
    # Points, optionally with error ellipses.
    def __init__(self, points, color = "red", marker_size = 10, ellipses = []):
        self.points = points
        self.color = color
        self.marker_size = marker_size
        self.ellipses = ellipses

    def background_draw(self):
        pass

    def draw(self, at_step):
        if at_step < len(self.points):
            # Draw points.
            pts = self.points[at_step]
            if len(pts) > 0:
                plt.scatter(pts[:,0], pts[:,1], color=self.color,
                            s=self.marker_size, zorder=10, linewidths=1,
                            edgecolors="black")

            # Draw ellipses, if present.
            if at_step < len(self.ellipses):
                eparams = self.ellipses[at_step]

                for i in range(min(len(pts), len(eparams))):
                    ell = matplotlib.patches.Ellipse(
                        pts[i], 2.0*eparams[i][1], 2.0*eparams[i][2],
                        np.rad2deg(eparams[i][0]),
                        color=self.color, fill=False, linewidth=1, zorder=20)
                    plt.gca().add_patch(ell)


# Particles are like points but add a direction vector.
class Particles(DrawableObject):
    def __init__(self, particles, color = "red", marker_size = 4.0,
                 vector_length = 50.0):
        self.particles = particles
        self.color = color
        self.marker_size = marker_size
        self.vector_length = vector_length

    def background_draw(self):
        pass

    def draw(self, at_step):
        if at_step < len(self.particles):
            pts = self.particles[at_step]
            # Draw one disk per particle.
            plt.scatter(pts[:,0], pts[:,1], color=self.color,
                        s=self.marker_size, zorder=20)
            # Draw a short line at each particle indicating the heading.
            xt = pts[:,0] + np.cos(pts[:,2]) * self.vector_length
            yt = pts[:,1] + np.sin(pts[:,2]) * self.vector_length
            l = np.vstack((pts[:,0], pts[:,1], xt, yt)).T
            plt.gca().add_collection(matplotlib.collections.LineCollection(
                l.reshape(l.shape[0], 2, 2), linewidths=0.5,
                colors = self.color, zorder=20))


# Main class.
class IPYLogfileViewer(object):
    def __init__(self, files=None, continuous_update=False,
                 matplotlib_format="svg"):
        # Ipy canvas extents in pixels, matplotlib extents in inches,
        # and matplotlib dpi. You may play with these to adjust appearance.
        # Note the result also depends on pixel or vector output, as selected
        # by the matplotlib_format parameter.
        self.canvas_width = 600
        self.plot_extents = (5, 5)
        self.plot_dpi = 100

        # World extents (in millimeters).
        self.world_extents = (2050.0, 2050.0)

        # The maximum scanner range used to scale scan drawings.
        self.max_scanner_range = 1900.0

        # Whether the slider will do continuous updates.
        self.continuous_update = continuous_update

        # Construct logfile (will be read in load_data()).
        self.logfile = LegoLogfile()

        # The lists of objects to draw.
        self.world_objects = []
        self.sensor_objects = []

        # Output format which is used by matplotlib.
        # Useful options: 'png' or 'svg'.
        set_matplotlib_formats(matplotlib_format)
        
        # Init GUI.
        self.init_ipy()

        # Trigger load.
        if files:
            self.load(files)

    def init_ipy(self):
        self.slider = widgets.IntSlider(min=0, max=0, value=0,
            continuous_update=self.continuous_update, description="Step",
            layout=widgets.Layout(width='95%'))
        self.world_widget  = widgets.interactive_output(
            self.draw_world, {'step': self.slider})
        self.sensor_widget = widgets.interactive_output(
            self.draw_sensor, {'step': self.slider})
        self.world_widget.layout = widgets.Layout(
            width=("%dpx" % self.canvas_width))
        self.sensor_widget.layout = widgets.Layout(
            width=("%dpx" % self.canvas_width))
        self.numeric_data = widgets.interactive_output(
            self.update_numeric_data, {'step': self.slider})

        self.hbox = widgets.HBox(
            children = (self.world_widget, self.sensor_widget))
        self.vbox = widgets.VBox(
            children = (self.hbox, self.numeric_data, self.slider))

        display(self.vbox)

    def draw_world(self, step):
        # Setup plot.
        fig, ax = plt.subplots(figsize=self.plot_extents, dpi=self.plot_dpi)
        ax.set_xlim(0, self.world_extents[0])
        ax.set_ylim(0, self.world_extents[1])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect("equal")

        # Call plot functions of objects.
        for d in self.world_objects:
            d.background_draw()
        for d in self.world_objects:
            d.draw(step)

    def draw_sensor(self, step):
        # Setup plot.
        fig, ax = plt.subplots(figsize=self.plot_extents, dpi=self.plot_dpi)
        ax.set_xlim(-self.max_scanner_range, self.max_scanner_range)
        # We move the center of the local coordinate system to the lower
        # half of the plot to have more space.
        ax.set_ylim(
            -self.max_scanner_range / 2, self.max_scanner_range * 3 / 2)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect("equal")

        # Draw the fixed parts.
        # TODO: hard-coded relative sizes and locations should be replaced
        # by something more general.
        ll = self.max_scanner_range * 0.9
        plt.gca().add_line(matplotlib.lines.Line2D( (-ll,0,0), (0, 0, ll),
            color="black", linewidth=1, zorder=20))
        plt.text(ll/20, ll, "x", zorder=20)
        plt.text(-ll, ll/20, "y", zorder=20)

        # Draw a disk in the scan center.
        plt.gca().add_patch(matplotlib.patches.Circle((0,0), radius=50,
                                                      color="gray", zorder=21))

        # Call plot functions of objects.
        for d in self.sensor_objects:
            d.background_draw()
        for d in self.sensor_objects:
            d.draw(step)

    def update_numeric_data(self, step):
        print(self.logfile.info(step))
     
    def load(self, all_file_names):
        for filename in all_file_names:
            self.logfile.read(filename)

        self.world_objects = []
        self.sensor_objects = []

        # Insert: landmarks.
        if self.logfile.landmarks:
            self.world_objects.append(Landmarks(self.logfile.landmarks))

        # Insert: reference trajectory.
        if self.logfile.reference_positions:
            self.world_objects.append(
                Trajectory(self.logfile.reference_positions,
                           cursor_color="red", background_color="#FFB4B4"))


        # Insert: scanner data.
        if self.logfile.scan_data:
            self.sensor_objects.append(ScannerData(self.logfile.scan_data))

        # Insert: detected cylinders, in scanner coord system.
        if self.logfile.detected_cylinders:
            # Need to rotate since sensor objects is x up, y left.
            rot = np.array([[0.0, -1.0], [1.0, 0.0]]).T
            cyl = [ (np.array(c) @ rot if len(c) > 0 else []) \
                    for c in self.logfile.detected_cylinders ]
            self.sensor_objects.append(Points(cyl, "#88FF88", marker_size=20))

        # Insert: world objects, cylinders and corresponding world objects,
        # ellipses.
        if self.logfile.world_cylinders:
            positions = [ np.array(cyl_one_scan) \
                          for cyl_one_scan in self.logfile.world_cylinders ]
            ellipses  = [ np.array(ell_one_scan) \
                          for ell_one_scan in self.logfile.world_ellipses ]
            self.world_objects.append(
                Points(positions, "#DC23C5", marker_size=30, ellipses=ellipses))

        # Insert: detected cylinders, transformed into world coord system.
        if self.logfile.detected_cylinders and \
           self.logfile.filtered_positions and \
           len(self.logfile.filtered_positions[0]) > 2:
            positions = []
            for i in range(min(len(self.logfile.detected_cylinders),
                               len(self.logfile.filtered_positions))):
                cyl = np.array(self.logfile.detected_cylinders[i])
                if cyl.size > 0:
                    pos = self.logfile.filtered_positions[i]
                    c, s = cos(pos[2]), sin(pos[2])
                    rot = np.array([[c, -s], [s, c]])
                    t_cyl = (rot @ cyl.T).T + pos[0:2]
                    positions.append(t_cyl)
                else:
                    positions.append([])
            self.world_objects.append(
                Points(positions, "#88FF88", marker_size=20))

        # Insert: particles.
        if self.logfile.particles:
            self.world_objects.append(
                Particles([np.array(pl) for pl in self.logfile.particles],
                          "#80E080"))

        # Insert: filtered trajectory.
        if self.logfile.filtered_positions:
            # If there is error ellipses, insert them as well.
            self.world_objects.append(
                Trajectory(self.logfile.filtered_positions,
                           standard_deviations=self.logfile.filtered_stddev,
                           cursor_color="blue", background_color="lightblue",
                           position_stddev_color = "#8080ff",
                           theta_stddev_color="#c0c0ff"))

        # Update slider.
        self.slider.max = max(0, self.logfile.size()-1)

        # Did not find a better way to trigger a redraw after load.
        # (Setting the value to 0 if it is already 0
        # does not trigger a redraw.)
        if self.logfile.size() > 1:
            self.slider.value = 1
            self.slider.value = 0

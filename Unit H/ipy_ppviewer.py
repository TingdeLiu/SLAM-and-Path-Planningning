# Viewer class for interactive plots in the SLAM lecture on path planning.
# Uses ipycanvas, ipywidgets, and ipyevents to provide the required
# functionality inside a jupyter notebook.
#
# In general, you should derive from this class, and implement compute().
# In some cases it may be necessary to implement __init_ as well.
# Example:
#
#    class MyPPViewer(PPViewer):
#        def compute(self, x1, y1, t1, x2, y2, t2, obstacles, potential):
#            # Only thing to do is to call the user-provided function,
#            # which returns (path, visited) tuple.
#            return astar((x1,y1), (x2,y2), obstacles+potential)
#    MyPPViewer(200,150, enable_potential_field=True).run()
#
# If override of init becomes necessary, use something like:
#
#        def __init__(self, width, height):
#            super().__init__(width, height, scale_factor=4,
#                enable_potential_field=True)
#            # ... additional part of __init__.
#
# (c) 14 JAN 2020 Claus Brenner
from ipycanvas import MultiCanvas, hold_canvas
from ipyevents import Event
from ipywidgets import AppLayout, Button, Checkbox, HBox, link
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

class PPViewer:
    def __init__(self, width, height, scale_factor = 3,
                 obstacle_stencil_halfwidth = 3,
                 draw_start_goal_heading = False,
                 enable_potential_field = False,
                 option_button_name = None):
        # Scale factor for display in jupyter notebook.
        self.scale_factor = int(scale_factor)
        # Half width of obstacle draw/undraw. Stencil size will be N2*2+1.
        self.obstacle_stencil_halfwidth = obstacle_stencil_halfwidth
        # If start/goal are to be drawn with a heading.
        self.draw_start_goal_heading = draw_start_goal_heading
        # If the potential field and the associated toggle button shall be used.
        self.enable_potential_field = enable_potential_field
        # Another optional toggle button, if a label for it is given.
        self.option_button_name = option_button_name

        # Background layers.
        self.obstacles = np.zeros((width, height), dtype=np.uint8)
        self.visited = np.zeros((width, height), dtype=np.uint8)        
        self.potential = np.zeros((width, height), dtype=np.uint8)
        
        # Canvas.
        self.canvas = MultiCanvas(2, width=width*self.scale_factor, height=height*self.scale_factor)
        
        # Events
        self.event = Event(source=self.canvas, watched_events=
            ['mousedown', 'mousemove', 'mouseup'])
        self.event.on_dom_event(self.handle_event)
        # Prevent default action showing right-click pop-up menu.
        Event(source=self.canvas, watched_events=['contextmenu'], 
            prevent_default_action=True)        
        # Mouse drag state: None or (x_start, y_start, button_number, shift_key).
        self.mousedrag = None
        
        # Elements that are drawn (on foreground).
        # Start and goal of the search. Either None or (x, y, dx, dy),
        # where dx dy is the unit vector of the heading.
        self.start_goal = [None, None]
        # Polyline path to draw.
        self.polyline_path = None

        # Additional widgets.
        self.clear_button = Button(description='Clear', tooltip='clear obstacles')
        self.clear_button.on_click(self.clear_obstacles)
        self.show_visited_button = Checkbox(value=True, description='Visited')
        self.show_visited_button.observe(self.checkbox_changed, names='value')
        self.show_potential_field_button = Checkbox(value=True, description='Potential')
        self.show_potential_field_button.observe(self.checkbox_changed, names='value')
        if option_button_name:
            self.option_button = Checkbox(value=False, description=option_button_name)
            self.option_button.observe(self.checkbox_changed, names='value')
        
        # Draw initial background.
        self.update_background()

    def clear_obstacles(self, _):
        self.obstacles.fill(0)
        self.potential.fill(0)
        self.compute_and_update()
        
    def checkbox_changed(self, _):
        self.compute_and_update()
        
    def option_button_state(self):
        return self.option_button.value

    def handle_event(self, event):
        et = event["type"]
        x, y = event["relativeX"], event["relativeY"]

        if et == "mousemove":
            if self.mousedrag is not None:
                self.mouse_update(x, y)

        elif et == "mousedown":
            if self.mousedrag is None:
                self.mousedrag = (x, y, event["button"], event["shiftKey"])
                if self.mousedrag[3]:  # set source/target.
                    sg = int(self.mousedrag[2] != 0)  # mouse button.
                    self.start_goal[sg] = (x, y, 1.0, 0.0)
                self.mouse_update(x, y)

        elif et == "mouseup":
            self.mousedrag = None
            self.compute_and_update()

    def mouse_update(self, x, y):
        if self.mousedrag is None: return
        
        # If shift key is pressed, it is source/target functionality.
        if self.mousedrag[3]:
            sg = int(self.mousedrag[2] != 0)
            sge = self.start_goal[sg]  # mouse button.
            dx, dy = x-sge[0], y-sge[1]
            if not (dx==0 and dy==0):
                r = np.sqrt(dx*dx + dy*dy)
                self.start_goal[sg] = (sge[0], sge[1], dx/r, dy/r)
            else:
                self.start_goal[sg] = (sge[0], sge[1], 1, 0)
            self.update_foreground()
        
        # Otherwise, it is set/remove obstacle functionality.
        else:
            xs, ys = x // self.scale_factor, y // self.scale_factor
            N = self.obstacle_stencil_halfwidth
            Ns = (N+0.5) * self.scale_factor
            if self.mousedrag[2] == 0:
                self.obstacles[xs-N:xs+N, ys-N:ys+N] = 255
                # Use foreground layer for a "quickdraw".
                self.canvas[1].fill_style = "#ff0000"
                self.canvas[1].fill_rect(x-Ns, y-Ns, 2*Ns, 2*Ns)
            else:
                self.obstacles[xs-N:xs+N, ys-N:ys+N] = 0
                self.canvas[1].fill_style = "#000000"
                self.canvas[1].fill_rect(x-Ns, y-Ns, 2*Ns, 2*Ns)
    
    def recompute_potential_field(self):
        # Maybe need to recompute potential field.
        if self.enable_potential_field and self.show_potential_field_button.value \
           and np.max(self.obstacles) > 0:
            # Compute distance transform.
            dist_transform = distance_transform_edt(255-self.obstacles)
            self.potential = np.maximum(255.0 - dist_transform*16, 0).astype(np.uint8) - self.obstacles
        else:
            self.potential = np.zeros_like(self.obstacles)
        
    def update_background(self):
        # Assemble image, zoom, and sent do canvas.
        z = np.zeros_like(self.visited, dtype=np.uint8)
        image = np.stack((self.obstacles.T,
            self.visited.T if self.show_visited_button.value else z.T,
            self.potential.T if self.enable_potential_field else z.T), axis=2)
        image = np.kron(image, np.ones((self.scale_factor, self.scale_factor, 1), dtype=np.uint8))
        self.canvas[0].put_image_data(image, 0, 0)
        
    def update_foreground(self):
        with hold_canvas(self.canvas[1]):
            self.canvas[1].clear()
            self.canvas[1].line_width = 3

            # Plot polyline path.
            if self.polyline_path is not None:
                self.canvas[1].stroke_style = "white"
                self.canvas[1].stroke_lines(self.polyline_path)
            # Plot start and goal node.
            for p, c in zip(self.start_goal, ("#ffff00", "#ff00ff")):
                if p is not None:
                    self.canvas[1].stroke_style = c
                    x, y, dx, dy = p
                    self.canvas[1].stroke_circle(x, y, 10)
                    if self.draw_start_goal_heading:
                        self.canvas[1].stroke_line(x,y,int(x+dx*30),(y+dy*30))

    def compute_and_update(self):
        self.recompute_potential_field()
        if self.start_goal[0] is not None and\
           self.start_goal[1] is not None:
            x1, y1, dx1, dy1 = self.start_goal[0]
            x2, y2, dx2, dy2 = self.start_goal[1]
            path, visited = self.compute(
                x1 // self.scale_factor, y1 // self.scale_factor, np.arctan2(dy1, dx1),
                x2 // self.scale_factor, y2 // self.scale_factor, np.arctan2(dy2, dx2),
                self.obstacles, self.potential)
            self.set_polyline_path(path)
            self.set_visited(visited)
        self.update_background()
        self.update_foreground()
            
    def set_polyline_path(self, path):
        self.polyline_path = [\
            ((x+0.5) * self.scale_factor, (y+0.5) * self.scale_factor + 0.5)\
            for x, y in path ]
        
    def set_visited(self, visited):
        # Rescale so that entire range is used.
        vabs = np.abs(visited)
        v_max = np.amax(vabs)
        if v_max > 0: vabs *= 255.0 / v_max
        self.visited = vabs.astype(np.uint8)     

    def compute(self, x1, y1, t1, x2, y2, t2, obstacles, potential):
        # Must return: (path, visited).
        pass  # To be implemented in user code.
    
    def run(self):
        b = (self.clear_button, self.show_visited_button)
        if self.enable_potential_field: b = b + (self.show_potential_field_button,)
        if self.option_button_name: b = b + (self.option_button,)
        return AppLayout(center=self.canvas, footer=HBox(b))

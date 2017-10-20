import traceback
from Tkinter import *
from PIL import ImageTk, Image
from functools import partial
import numpy as np

__author__ = 'mhuijser'


class VerticalScrolledFrame(Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling
    https://github.com/ewanbarr/anansi/blob/master/anansi/ui_tools/scrollable_frame.py
    """

    def __init__(self, parent, *args, **kw):
        Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Scrollbar(self, orient=VERTICAL)
        vscrollbar.pack(fill=Y, side=RIGHT, expand=False)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
            if interior.winfo_reqheight() != canvas.winfo_height():
                canvas.config(height=min(interior.winfo_reqheight(), 950))

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)

        def _bound_to_mousewheel(event):
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbound_to_mousewheel(event):
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        def _on_mousewheel(event):
            if event.num == 5:
                canvas.yview_scroll(-1, "units")
            elif event.num == 4:
                canvas.yview_scroll(1, "units")

        interior.bind('<Enter>', _bound_to_mousewheel)
        interior.bind('<Leave>', _unbound_to_mousewheel)


class PointLabelInterface(object):
    def __init__(self, point_query_image, classes, classes_dictionary):
        """
        Provides functions for point-labeling (traditional active learning) by human.
        :param point_query_image: the to-be-labeled query image.
        :param classes: list of integers denoting the possible classes.
        :param classes_dictionary: dictionary with key the class as int and value the class label as string.
        """
        self.point_query_image = point_query_image
        self.classes = classes
        self.classes_dictionary = classes_dictionary
        self.label_point_query = None
        self.master = Tk()
        self.make_image()
        self.master.mainloop()

    def make_image(self):
        self.image_frame = Frame()
        self.image_frame.grid(row=3, column=1, columnspan=1)

        self.confirm_button = Button(self.image_frame, command=self.confirm_callback, text="Confirm", state=DISABLED)
        self.confirm_button.grid(row=3, column=1)
        class_var = StringVar()
        self.dropdown_class = OptionMenu(self.image_frame, class_var, self.classes_dictionary[self.classes[0]],
                                         self.classes_dictionary[self.classes[1]], command=self.class_dropdown_callback)
        self.dropdown_class.var = class_var
        self.dropdown_class.grid(row=2, column=1)

        if self.point_query_image.shape[1] == 1:
            query_img = self.point_query_image[0, :, :].squeeze()
        else:
            query_img = self.point_query_image.squeeze().transpose(1, 2, 0)
        query_img = Image.fromarray(np.uint8(255 * query_img))
        query_img = ImageTk.PhotoImage(query_img)
        imglabel = Label(self.image_frame, image=query_img, text="point query", compound=BOTTOM)
        imglabel.image = query_img
        imglabel.grid(row=1, column=1)
        self.image_frame.pack()
        return True

    def class_dropdown_callback(self, value):
        self.value_dropdown = value
        self.update_confirm_button()

    def update_confirm_button(self):
        if self.dropdown_class.var.get() != "":
            self.confirm_button.configure(state=NORMAL)
        else:
            self.confirm_button.configure(state=DISABLED)

    def find_key(self, value):
        for key, val in self.classes_dictionary.items():
            if val == value:
                return key

    def confirm_callback(self):
        self.label_point_query = self.find_key(self.value_dropdown)
        self.image_frame.destroy()
        self.master.destroy()


class LabelInterface(object):
    def __init__(self, line_vectors, line_images, point_query, point_query_image, intersection_point_cdb, classes,
                 classes_dictionary):
        """
        Display query line and provide interface for human annotator to label it.
        :param line_vectors: points on query line.
        :param line_images: images generated from points on query line.
        :param point_query: the query sample, as point on line.
        :param point_query_image: the generated image corresponding to the query sample point.
        :param intersection_point_cdb: intersection point of query line with current estimated decision boundary.
        :param classes: list of integers denoting the possible classes.
        :param classes_dictionary: dictionary with key the class as int and value the class label as string.
        """
        self.ll = 14 # number of images.
        self.classes = classes
        self.classes_dictionary = classes_dictionary
        self.handbags = False
        if line_images.shape[2] > 40:
            pass
        self.line = line_vectors  # (n_samples, n_dim)
        self.line_images = line_images  # (n_samples, n_channels, height, width)
        self.point_query = point_query  # (n_dim,1)
        self.point_query_image = point_query_image
        self.intersection_point_cdb = intersection_point_cdb
        self.__selected = None
        self.select_colour = "forest green"
        self.base_colour = 'LightSkyBlue1'
        self.intersection_colour = "forest green"
        self.query_colour = "DarkOrange2"
        self.master = Tk()
        self.line_index = 0
        self.t_values_buttons = None
        self.t_values_images = None
        self.label_point_query = None
        self.chosen_point = None
        self.results_dictionary = {}
        try:
            self.make_line(self.line_images)
        except Exception as e:
            print traceback.print_exc()
        self.master.mainloop()

    def compute_length(self, line_segment):
        return np.linalg.norm(self.to_vector(line_segment[0, :]) - self.to_vector(line_segment[-1, :]))

    def compute_distance(self, p1, p2):
        print p1.shape, p2.shape
        return np.linalg.norm(p1 - p2)

    def to_vector(self, x):
        try:
            if x.shape[1] == 1:
                return x
            elif x.shape[0] == 1 and x.shape[1] > 1:
                return x.reshape((x.shape[1], 1))
            else:
                raise TypeError("This is not a vector!")
        except IndexError:
            return x.reshape((len(x), 1))

    def compute_t(self, A, B, point):
        AB = B - A
        return np.asscalar(np.dot((point - A).T, (AB)) / np.dot((AB).T, (AB)))

    def define_line(self, line_segment):
        A = self.to_vector(line_segment[0, :])
        B = self.to_vector(line_segment[-1, :])
        # A + (B-A)t
        return A, B

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx

    def set_intersection_point(self, t):
        button = self.get_button_for_t(t)
        button.config(highlightbackground=self.intersection_colour, highlightthickness=1.5)

    def set_query_point(self, t):
        button = self.get_button_for_t(t)
        button.config(highlightbackground=self.query_colour, highlightthickness=1.5)

    def make_line(self, line_images):
        self.t_values_buttons = []
        self.t_values_images = []
        self.A, self.B = self.define_line(self.line)
        # print "Length", self.compute_length(self.line)
        t_intersection_point = self.compute_t(self.A, self.B, self.intersection_point_cdb)
        t_point_query = self.compute_t(self.A, self.B, self.point_query)

        self.scrolled_window = VerticalScrolledFrame(self.master)
        self.line_frame = Frame(self.scrolled_window.interior)
        self.buttons = []

        line_length = self.line.shape[0]
        row = -1
        self.line_frame.grid(row=line_length / self.ll, column=29, columnspan=1)
        col_counter = 0
        # Display it
        for i, image in enumerate(line_images):
            col = i % self.ll
            if col == 0:
                col_counter = 0
                row += 1
            t = self.compute_t(self.A, self.B, self.to_vector(self.line[i, :]))
            self.t_values_images.append(t)
            if image.shape[0] == 1:
                img = image[0, :, :].squeeze()
            else:
                img = image.transpose(1, 2, 0).squeeze()
            img = Image.fromarray(np.uint8(255 * img))
            img = ImageTk.PhotoImage(img)
            callback = partial(self.click_callback, i)
            button = Button(self.line_frame, command=callback, bg=self.base_colour)
            button.grid(row=row, column=col_counter)
            col_counter += 1
            self.buttons.append(button)

            imglabel = Label(self.line_frame, image=img, text="%.2f" % t, compound=BOTTOM)
            imglabel.image = img
            imglabel.grid(row=row, column=col_counter)
            col_counter += 1
        callback = partial(self.click_callback, line_length)
        last_button = Button(self.line_frame, command=callback, bg=self.base_colour)
        self.buttons.append(last_button)
        last_button.grid(row=row, column=col_counter)
        self.line_frame.columnconfigure(13, weight=1)

        step = 0.5 * (self.t_values_images[1] - self.t_values_images[0])
        self.t_values_buttons = np.linspace(self.t_values_images[0] - step, self.t_values_images[-1] + step,
                                            len(self.t_values_images) + 1)
        self.set_intersection_point(t_intersection_point)
        self.set_query_point(t_point_query)

        r = line_length / self.ll + 3
        self.buttonframe = Frame(self.scrolled_window.interior)
        self.buttonframe.grid(row=r, column=3, columnspan=29)
        self.confirm_button = Button(self.buttonframe, command=self.confirm_callback, text="Confirm", state=DISABLED)
        self.confirm_button.grid(row=r, column=1)
        Label(self.buttonframe, text="Proper line:", font=("Helvetica", 12)).grid(row=r - 1, column=0)
        yes_var = IntVar()
        yes_callback = partial(self.update_checkbuttons, "Yes")
        self.checkbox_yes = Checkbutton(self.buttonframe, text="Yes", variable=yes_var, command=yes_callback)
        self.checkbox_yes.var = yes_var
        self.checkbox_yes.grid(row=r - 1, column=1)
        no_var = IntVar()
        no_callback = partial(self.update_checkbuttons, "No")
        self.checkbox_no = Checkbutton(self.buttonframe, text="No", variable=no_var, command=no_callback)
        self.checkbox_no.var = no_var
        self.checkbox_no.grid(row=r - 1, column=2)
        class_var = StringVar()
        self.dropdown_class = OptionMenu(self.buttonframe, class_var, self.classes_dictionary[self.classes[0]],
                                         self.classes_dictionary[self.classes[1]], command=self.class_dropdown_callback)
        self.dropdown_class.var = class_var
        self.dropdown_class.grid(row=r - 2, column=1)
        if self.point_query_image.shape[1] == 1:
            query_img = self.point_query_image[0, :, :].squeeze()
        else:
            query_img = self.point_query_image.squeeze().transpose(1, 2, 0)
        query_img = Image.fromarray(np.uint8(255 * query_img))
        query_img = ImageTk.PhotoImage(query_img)
        imglabel = Label(self.buttonframe, image=query_img, text="point query", compound=BOTTOM)
        imglabel.image = query_img
        imglabel.grid(row=r - 2, column=2)
        self.scrolled_window.pack()
        return True

    def class_dropdown_callback(self, value):
        self.value_dropdown = value
        self.update_confirm_button()

    def update_checkbuttons(self, checkbutton_label):
        if checkbutton_label == "Yes":
            if self.checkbox_yes.var.get():
                self.checkbox_no.deselect()
        elif checkbutton_label == "No":
            if self.checkbox_no.var.get():
                self.checkbox_yes.deselect()
        self.update_confirm_button()

    def update_confirm_button(self):
        if self.checkbox_no.var.get() and self.dropdown_class.var.get() != "":
            self.confirm_button.configure(state=NORMAL)
        elif self.one_checkbutton_true() and self.selected is not None and self.dropdown_class.var.get() != "":
            self.confirm_button.configure(state=NORMAL)
        else:
            self.confirm_button.configure(state=DISABLED)

    def one_checkbutton_true(self):
        return self.checkbox_yes.var.get() or self.checkbox_no.var.get()

    @property
    def selected(self):
        return self.__selected

    @selected.setter
    def selected(self, number):
        self.__selected = number
        self.update_confirm_button()

    def click_callback(self, button_number):
        for b in self.buttons:
            b.configure(bg=self.base_colour)
        if button_number == self.selected:
            self.selected = None
        else:
            self.buttons[button_number].configure(bg=self.select_colour)
            self.selected = button_number

    def get_vector_from_button_number(self, line_index, button_number):
        line = lambda t: self.A + (self.B - self.A) * t
        vector = line(self.t_values_buttons[button_number])
        return vector

    def get_button_for_t(self, t):
        _, idx = self.find_nearest(self.t_values_buttons, t)
        return self.buttons[idx]

    def process_result(self, current_line_index, button_number, value_seebothclasses):
        if value_seebothclasses and button_number is not None:
            self.chosen_point = self.get_vector_from_button_number(current_line_index, button_number)
        else:
            self.chosen_point = None
        self.label_point_query = self.find_key(self.value_dropdown)

    def find_key(self, value):
        for key, val in self.classes_dictionary.items():
            if val == value:
                return key

    def confirm_callback(self):
        self.process_result(self.line_index, self.selected, self.checkbox_yes.var.get())
        self.selected = None
        self.line_frame.destroy()
        self.scrolled_window.interior.destroy()
        self.scrolled_window.destroy()
        self.master.destroy()

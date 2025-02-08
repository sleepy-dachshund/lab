import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

COLOR_DICT = {'blue1': (0, 44, 86), 'blue2': (0, 66, 128), 'blue3': (43, 151, 255),
              'orange1': (255, 102, 50), 'orange2': (153, 39, 0), 'orange3': (229, 58, 0),
              'gold1': (195, 148, 40), 'gold2': (97, 74, 20), 'gold3': (146, 111, 30),
              'green1': (51, 151, 111), 'green2': (25, 75, 55), 'green3': (38, 113, 83),
              'purple1': (120, 118, 208), 'purple2': (43, 42, 121), 'purple3': (65, 62, 182),
              'red1': (162, 28, 54), 'red2': (81, 14, 27), 'red3': (122, 21, 40)}


def color_input_formatted(color):
    if color in COLOR_DICT:
        return tuple([x / 255 for x in COLOR_DICT[color]])
    else:
        return color


blue1 = colors.Color(*color_input_formatted('blue1'))
blue3 = colors.Color(*color_input_formatted('blue3'))
red1 = colors.Color(*color_input_formatted('red1'))
red3 = colors.Color(*color_input_formatted('red3'))


def get_color_from_cmap(norm, cmap_name='coolwarm'):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)  # returns (r, g, b, a)
    return colors.Color(rgba[0], rgba[1], rgba[2])


class PDFReport:
    def __init__(self, filename, page_size=letter):
        self.filename = filename
        self.page_size = page_size
        self.width, self.height = page_size
        self.c = canvas.Canvas(self.filename, pagesize=self.page_size)
        self.pages = []

    def add_page(self, header_text=None, footer_text=None, content_fn=None):
        """
        Adds a new page. The content_fn is a function that takes the canvas as an input
        and draws the main content (tables, images, etc.) on the canvas.
        :param header_text:
        :param footer_text:
        :param content_fn:
        :return:
        """
        if header_text:
            self.add_header(header_text)
        if footer_text:
            self.add_footer(footer_text)
        if content_fn is not None:
            content_fn(self.c, self.width, self.height)
        self.c.showPage()

    def add_header(self, text, banner_height=35, font='Helvetica-Bold', font_size=16, fill_color=blue1):
        """
        Draw a header at top center
        :param text:
        :param banner_height:
        :param font:
        :param font_size:
        :param fill_color:
        :return:
        """
        # Draw the banner rectangle at the top
        self.c.setFillColor(fill_color)
        self.c.rect(0, self.height - banner_height, self.width, banner_height, fill=1, stroke=0)

        # Set white text and desired font
        self.c.setFont(font, font_size)
        self.c.setFillColor(colors.white)

        # Calculate horizontal centering of the text
        text_width = self.c.stringWidth(text, font, font_size)
        x_text = (self.width - text_width) / 2

        # Approximate vertical centering within the banner
        y_text = self.height - banner_height / 2 - font_size / 4

        self.c.drawString(x_text, y_text, text)

    def add_footer(self, text, y_offset=20, font='Helvetica', font_size=10, fill_color=blue1):
        """
        Draw a footer at bottom center
        :param text:
        :param y_offset:
        :param font:
        :param font_size:
        :param fill_color:
        :return:
        """
        self.c.setFont(font, font_size)
        self.c.setFillColor(fill_color)
        test_width = self.c.stringWidth(text, font, font_size)
        self.c.drawString((self.width - test_width) / 2, y_offset / 2, text)

    def add_matplotlib_figure(self, fig, x=50, y=300, width=400, height=250):
        """
        Saves a matplotlib to a temporary file and adds it as an image to the canvas.
        :param fig:
        :param x:
        :param y:
        :param width:
        :param height:
        :return:
        """
        temp_image = 'temp_plot.png'
        fig.savefig(temp_image, bbox_inches='tight', dpi=100)
        self.c.drawImage(temp_image, x, y, width=width, height=height)
        os.remove(temp_image)

    def add_table(self, df, x=50, y=250, col_widths=None, row_heights=None, config=None):
        """
        Create a table with the given pandas df with optional configuration.
        The config dictionary can contain keys like:
            - 'merge_cells': list of tuples (start, end) e.g., [((col1, row1), (col2, row2)), ...]
            - 'conditional_format': list of dicts, e.g.
                [{'cells': (col, row), 'bg_color': colors.lightblue, 'text_color': colors.black},  ...]
            - 'general_style': list of base style commands to apply
        :param df:
        :param x:
        :param y:
        :param col_widths:
        :param row_heights:
        :param config:
        :return:
        """
        # check if the df has a multi-index, if so, add index values as columns
        if isinstance(df.index, pd.MultiIndex):
            # prepare header: the index level names will be set and then the dataframe columns
            idx_names = list(df.index.names)
            header = idx_names + df.columns.tolist()
            data = [header]
            # append each row: convert index tuple to list then extend with row values
            for idx, row in zip(df.index, df.values):
                if not isinstance(idx, tuple):
                    idx = (idx,)
                data.append(list(idx) + list(row))
        else:
            # convert df to list of lists (including header)
            data = [df.columns.tolist()] + df.values.tolist()

        # define default column widths and row heights if not provided
        n_cols = len(data[0])
        if col_widths is None:
            col_widths = [self.width / n_cols] * n_cols
        if row_heights is None:
            row_heights = [25] * len(data)

        table = Table(data, colWidths=col_widths, rowHeights=row_heights)
        style = TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 10)
        ])

        if isinstance(df.index, pd.MultiIndex):
            n_index = len(df.index.names)
            style.add('BACKGROUND', (0, 0), (n_index - 1, 0), blue1)
            style.add('TEXTCOLOR', (0, 0), (n_index - 1, 0), colors.white)
            style.add('BACKGROUND', (n_index, 0), (-1, 0), blue1)
            style.add('TEXTCOLOR', (n_index, 0), (-1, 0), colors.white)
        else:
            style.add('BACKGROUND', (0, 0), (-1, 0), blue1)
            style.add('TEXTCOLOR', (0, 0), (-1, 0), colors.white)

        # merge cells for repeated multi-index values and shade in grey
        if isinstance(df.index, pd.MultiIndex):
            n_index = len(df.index.names)
            # for each index column, scan the table rows and merge cells with same value
            for col in range(n_index):
                start_row = 1
                current_val = data[start_row][col]
                for row in range(2, len(data)):
                    if data[row][col] == current_val:
                        continue
                    else:  # if more than one row has the same value, merge them
                        if row - start_row > 1:
                            style.add('SPAN', (col, start_row), (col, row - 1))
                        style.add('BACKGROUND', (col, start_row), (col, row - 1), colors.lightgrey)
                        current_val = data[row][col]
                        start_row = row
                # check for merge at end of table
                if len(data) - start_row > 1:
                    style.add('SPAN', (col, start_row), (col, len(data) - 1))
                style.add('BACKGROUND', (col, start_row), (col, len(data) - 1), colors.lightgrey)

        if config:
            # apply merging if provided
            if 'merge_cells' in config:
                for merge_range in config['merge_cells']:
                    start, end = merge_range
                    style.add('SPAN', start=start, end=end)

            # apply conditional formatting if provided
            if 'conditional_format' in config:
                for fmt in config['conditional_format']:
                    cells = fmt.get('cells')
                    bg = fmt.get('bg_color', None)
                    text_color = fmt.get('text_color', None)
                    if bg:
                        style.add('BACKGROUND', cells, cells, bg)
                    if text_color:
                        style.add('TEXTCOLOR', cells, cells, text_color)

            # additional general style commands
            if 'general_style' in config:
                for rule in config['general_style']:
                    style.add(*rule)

        table.setStyle(style)

        # draw table.
        # note: Table.wrap returns the width, height of the table
        # the provided y here is intended as a top reference point for the table
        # , and since ReportLab draws objects from the bottom left corner,
        # we need to adjust the y position so that the table is shifted downward by its own height 'th'
        # so that it's top aligns with the y provided
        tw, th = table.wrap(self.width, self.height)
        table.drawOn(self.c, x, y - th)

    def save(self):
        self.c.save()


def content_fn_grid(c, width, height):
    # Layout parameters
    margin_top, margin_bottom = 50, 50
    margin_left, margin_right = 50, 50
    spacer_x, spacer_y = 10, 10
    n_rows, n_cols = 3, 2  # adjust as needed (e.g. 1x2 for 2 items, 2x3 for 6 items)

    # Compute available area
    avail_width = width - margin_left - margin_right - (n_cols - 1) * spacer_x
    avail_height = height - margin_top - margin_bottom - (n_rows - 1) * spacer_y
    cell_w = avail_width / n_cols
    cell_h = avail_height / n_rows

    # List of items to place (order: row-major from top left)
    items = [
        {'type': 'plot', 'content': fig},
        {'type': 'plot', 'content': fig},
        {'type': 'plot', 'content': fig},
        {'type': 'plot', 'content': fig},
        {'type': 'plot', 'content': fig},
        {'type': 'plot', 'content': fig},
        # Add more items as needed...
    ]
    if len(items) > n_rows * n_cols:
        raise ValueError(f"Too many items for the grid layout in content_fn_grid(): {len(items)} items but only {n_rows} * {n_cols} cells available.")

    for idx, item in enumerate(items):
        row = idx // n_cols
        col = idx % n_cols
        x = margin_left + col * (cell_w + spacer_x)
        # y_top: top of the cell (pages origin is bottom left)
        y_top = height - margin_top - row * (cell_h + spacer_y)
        if item['type'] == 'plot':
            # For plots, specify y as lower-left corner (y_top - cell height)
            pdf.add_matplotlib_figure(item['content'], x=x, y=y_top - cell_h, width=cell_w, height=cell_h)
        elif item['type'] == 'table':
            # For tables, our add_table uses y as the top reference
            pdf.add_table(item['content'], x=x, y=y_top)


def content_fn_custom(c, width, height):
    # Define boxes manually. Each box specifies its position and size.
    boxes = [
        # Top row: two plots side-by-side
        {
            'type': 'plot',
            'content': fig,
            'x': 50,
            'y': height - 100,  # top of the box
            'width': (width - 100 - 10) / 2,
            'height': 200
        },
        {
            'type': 'plot',
            'content': fig,
            'x': 50 + ((width - 100 - 10) / 2) + 10,
            'y': height - 100,
            'width': (width - 100 - 10) / 2,
            'height': 200
        },
        # Bottom row: one table spanning full width
        {
            'type': 'table',
            'content': simple_df,
            'x': 50,
            'y': height / 2  # top of the table
            # you can optionally add 'col_widths' or 'row_heights' here
        }
    ]

    for box in boxes:
        if box['type'] == 'plot':
            pdf.add_matplotlib_figure(box['content'],
                                      x=box['x'],
                                      y=box['y'] - box['height'],
                                      width=box['width'],
                                      height=box['height'])
        elif box['type'] == 'table':
            page_width = width - 100
            col_widths = [page_width / len(box['content'].columns)] * len(box['content'].columns)
            pdf.add_table(box['content'], x=box['x'], y=box['y'], col_widths=col_widths)


def get_dynamic_rules(df: pd.DataFrame, column_idx: int):
    # Determine how many extra index columns exist (if using a MultiIndex)
    num_index = len(df.index.names) if isinstance(df.index, pd.MultiIndex) else 0
    # Our target is overall column index 3 (4th column); so the corresponding data column is:
    target_data_col = column_idx - num_index

    # Extract the values from the target data column in the dataframe
    col_values = df.iloc[:, target_data_col]
    min_val = col_values.min()
    max_val = col_values.max()
    diff = max_val - min_val if max_val != min_val else 1

    # Build dynamic conditional formatting rules for each data row (table row 0 is header)
    dynamic_rules = []
    for i, value in enumerate(col_values, start=1):
        norm = (value - min_val) / diff
        bg_color = get_color_from_cmap(norm)
        dynamic_rules.append({
            'cells': (column_idx, i),  # 4th column, row i
            'bg_color': bg_color,
            'text_color': colors.black
        })
    return dynamic_rules


if __name__ == "__main__":
    # create simple dataframe
    simple_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15]
    })

    # create multi-index dataframe
    arrays = [['Group1', 'Group1', 'Group2', 'Group2'],
              ['SubA', 'SubB', 'SubA', 'SubB']]
    index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=['Group', 'Subgroup'])
    multi_df = pd.DataFrame({
        'Value1': [10, 20, 30, 40],
        'Value2': [50, 60, 70, 80]
    }, index=index)

    # create a sample matplotlib line plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], label='y = x^2', color=color_input_formatted('green1'))
    ax.plot([1, 2, 3, 4, 5], [1, 8, 27, 64, 125], label='y = x^3', color=color_input_formatted('red1'))
    ax.set_title('Sample Line Plot')
    ax.legend()

    # ========== CREATE A PDF REPORT ========== #
    pdf = PDFReport('sample_report.pdf')

    '''
    # NOTE ON POSITIONING AND SIZING:
    # todo: note for ReportLab, the origin (0, 0) is at the bottom left corner of the page.
    #  -- the X coordinate increases to the right, and the Y coordinate increases upwards. 
    #  -- so positions are defined relative to the bottom left corner of the page. 
    #  -- and each object's lower left corner is placed at the specified position relative to that corner.
    
    # todo: width is the width (in points) that the image will occupy on the canvas 
    #  -- height is the height (in points) that the image will occupy on the canvas. 
    #  -- the image will be scaled (if necessary) so that it fits exactly within a rectangle of (width x height) points.
    
    PLOT EXAMPLE:
    
    pdf.add_matplotlib_figure(fig, x=50, y=300, width=400, height=250)
    - the lower left corner of the image will be 50 points from the left edge of the page,
    - and 300 points from the bottom edge of the page.
    - the image will be drawn in a rectangle that is 400 points wide and 250 points tall.
    
    TABLE EXAMPLE:
    pdf.add_table(simple_df, x=50, y=250)
    -- x and y are the coordinates of the lower left corner of the table, relative to the bottom left corner of the page.
    -- col_widths: an optional list specifying the width (in points) of each column in the table.
        - if not provided, the code defaults to equal widths based on total page width.
    -- row_heights: an optional list specifying the height (in points) of each row in the table.
        - if not provided, the code defaults to a fixed height in points for each row.
    -- config: an optional dictionary with additional configuration options for the table.
        
    
    '''

    # ========== PAGE 1: simple df and matplotlib plot ========== #
    def content_page1(c, width, height):
        page_width = width - 100
        plot_height = 300
        bottom_plot = height - plot_height - 50

        pdf.add_matplotlib_figure(fig, x=50, y=bottom_plot, width=page_width, height=plot_height)
        col_widths = [page_width / len(simple_df.columns)] * len(simple_df.columns)
        pdf.add_table(simple_df, x=50, y=bottom_plot - 50, col_widths=col_widths)

    pdf.add_page(header_text='Sample Report - Page 1', footer_text='Page 1', content_fn=content_page1)

    # ========== PAGE 2: multi-index df and table with a customized table formatting ========== #
    def content_page2(c, width, height):
        dynamic_rules = get_dynamic_rules(multi_df, 3)  # note column_idx is inclusive of index columns
        config = {
            'merged_cells': [((0, 0), (1, 0))],
            'conditional_format': [
                {'cells': (1, 1), 'bg_color': colors.lightgrey, 'text_color': red1},
                {'cells': (2, 2), 'bg_color': red3, 'text_color': blue1}
            ] + dynamic_rules,
            'general_style': [
                ('FONTSIZE', (0, 0), (-1, -1), 10)
            ]
        }
        page_width = width - 100
        num_cols = len(multi_df.columns) + len(multi_df.index.names)
        col_widths = [page_width / num_cols] * num_cols
        pdf.add_table(multi_df, x=50, y=height-100, config=config, col_widths=col_widths)

    pdf.add_page(header_text='Sample Report - Page 2', footer_text='Page 2', content_fn=content_page2)

    # ========== PAGE 3: grid layout with multiple plots ========== #
    pdf.add_page(header_text='Sample Report - Page 3', footer_text='Page 3', content_fn=content_fn_grid)

    # ========== PAGE 4: custom layout with plots and tables ========== #
    pdf.add_page(header_text='Sample Report - Page 4', footer_text='Page 4', content_fn=content_fn_custom)

    pdf.save()
    print(f'PDF report saved to {pdf.filename}')


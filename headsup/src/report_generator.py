from datetime import datetime
import pandas as pd


class ReportGenerator:
    def __init__(self, ic, ir, sectors, sc, sr, screener, screener_desc):
        self.index_characteristics = ic
        self.index_returns = ir
        self.sector_characteristics = sectors
        self.select_characteristics = sc
        self.select_returns = sr
        self.screener = screener
        self.screener_desc = screener_desc

    def generate_html_report(self):
        now = datetime.now().strftime('%Y-%m-%d')

        # add tables
        html = f"""
        <html>
        <body>
        <h2>Daily Heads Up Report - {now}</h2>

        <h3>Recent Market Returns</h3>
        {self.index_returns.to_html(float_format=lambda x: '%.2f' % x)}
        
        <h3>Market Characteristics</h3>
        {self.index_characteristics.to_html(float_format=lambda x: '%.2f' % x)}
        
        <h3>Sector Characteristics</h3>
        {self.sector_characteristics.to_html(float_format=lambda x: '%.2f' % x)}
        
        <h3>Select Characteristics</h3>
        {self.select_characteristics.to_html(float_format=lambda x: '%.2f' % x)}

        <h3>Select Returns</h3>
        {self.select_returns.to_html(float_format=lambda x: '%.2f' % x)}
        
        <h3>Screener Results</h3>
        {self.screener.to_html(float_format=lambda x: '%.2f' % x)}
        
        <h3>Screener Descriptions</h3>
        {self.screener_desc.to_html(float_format=lambda x: '%.2f' % x)}
        
        
        <ul>
        """

        # # add plots
        # html += """
        # <h3>Plots</h3>
        # """
        #
        # html += """
        # </ul>
        # </body>
        # </html>
        # """

        return html
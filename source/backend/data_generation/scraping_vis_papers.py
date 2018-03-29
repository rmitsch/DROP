"""
Note: Hastily written script for scraping data from keyvis.org and IEEE. Don't expect quality code here.
"""

import scrapely
import html2text
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import requests
from bs4 import BeautifulSoup

########################################
# !. Get cluster information from
# keyvis.
########################################

cluster_name_scraper = scrapely.Scraper()
url = "http://keyvis.org/expert.php"
data = {'panel': 'panel-group'}
cluster_name_scraper.train(url, data)


h = html2text.HTML2Text()
# Crawl page.
content = cluster_name_scraper.scrape(url)[0]["panel"][0]
# Extract useful content from HTML.
parsed_content = h.handle(content)
# Split up in individual elements.
parsed_content_list = parsed_content.strip().split("####")

########################################
# 2. Get detailed information on papers
# in clusters.
########################################

# Train crawler on information on IEEE page (since apparently all papers are linked there).
paper_scraper = scrapely.Scraper()
url = "http://ieeexplore.ieee.org/document/7194853/"
data = {
    'title': 'Mining Graphs for Understanding Time-Varying Volumetric Data',
    'abstract': 'A notable recent trend in time-varying volumetric data analysis and visualization is to extract data relationships and represent them in a low-dimensional abstract graph view for visual understanding and making connections to the underlying data. Nevertheless, the ever-growing size and complexity of data demands novel techniques that go beyond standard brushing and linking to allow significant reduction of cognition overhead and interaction cost. In this paper, we present a mining approach that automatically extracts meaningful features from a graph-based representation for exploring time-varying volumetric data. This is achieved through the utilization of a series of graph analysis techniques including graph simplification, community detection, and visual recommendation. We investigate the most important transition relationships for time-varying data and evaluate our solution with several time-varying data sets of different sizes and characteristics. For gaining insights from the data, we show that our solution is more efficient and effective than simply asking users to extract relationships via standard interaction techniques, especially when the data set is large and the relationships are complex. We also collect expert feedback to confirm the usefulness of our approach.'
}
paper_scraper.train(url, data)


# Loop through clusters, aggregate information.
for i, cluster_title in enumerate(parsed_content_list):
    if cluster_title != "":
        clean_cluster_title = cluster_title.strip()

        # Get detailled paper information.
        response = urlopen(
            Request(
                'http://keyvis.org/load3.php',
                urlencode({'expert': str(i)}).encode()
            )
        ).read().decode()
        parsed_paper_response = h.handle(response)

        # Dismiss keywords at beginning, only keep from first paper on.
        parsed_paper_response = parsed_paper_response[
            parsed_paper_response.index("---|---|---") + 11:
        ]

        # Split by ---|---|--- to get information on individual papers.
        parsed_paper_list = parsed_paper_response.strip().split("---|---")

        # Loop through papers, collect information on papers.
        for paper in parsed_paper_list:
            paper_info = {
                "cluster_title": clean_cluster_title,
                "conference": None,
                "year": None,
                "title": None,
                "abstract": None,
                "url": None
            }

            # Split paper in description to extract conference, year and link.
            paper_parts = paper.split("|")

            if len(paper_parts) > 1:
                paper_info["conference"] = paper_parts[0].strip()
                paper_info["year"] = paper_parts[1].strip()

                # For URL to paper: Find last occurence of round bracket, read URL except closing round bracket).
                paper_info["url"] = paper_parts[2][paper_parts[2].rfind('(') + 1:-2]
                content = paper_scraper.scrape(paper_info["url"])[0]
                # Get paper title.
                paper_info["title"] = h.handle(content["title"][0]) \
                    .replace('\n', ' ') \
                    .replace('\r', '') \
                    .replace(" - IEEE Journals & Magazine", "") \
                    .strip()
                # Get abstract.
                abstract = h.handle(content["abstract"][0]) \
                    .replace('\n', ' ') \
                    .replace('\r', '') \
                    .strip()
                abstract_index = abstract.index('"abstract":') + 12
                abstract_end_index = abstract_index + (abstract[abstract_index + 1:].index('","')) + 1
                paper_info["abstract"] = abstract[abstract_index:abstract_end_index]

            if paper_info["abstract"] is not None:
                print(paper_info)
        exit()

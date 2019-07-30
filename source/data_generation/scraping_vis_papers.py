"""
Note: Hastily written script for scraping data from keyvis.org and IEEE. Don't expect quality code here.
"""

import scrapely
import html2text
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import urllib
import csv
from collections import Set

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

# # Train crawler on information on IEEE page (since apparently all papers are linked there).
# paper_scraper = scrapely.Scraper()
# url = "http://ieeexplore.ieee.org/document/7194853/"
# data = {
#     'title': 'Mining Graphs for Understanding Time-Varying Volumetric Data',
#     'abstract': 'A notable recent trend in time-varying volumetric data analysis and visualization is to extract data relationships and represent them in a low-dimensional abstract graph view for visual understanding and making connections to the underlying data. Nevertheless, the ever-growing size and complexity of data demands novel techniques that go beyond standard brushing and linking to allow significant reduction of cognition overhead and interaction cost. In this paper, we present a mining approach that automatically extracts meaningful features from a graph-based representation for exploring time-varying volumetric data. This is achieved through the utilization of a series of graph analysis techniques including graph simplification, community detection, and visual recommendation. We investigate the most important transition relationships for time-varying data and evaluate our solution with several time-varying data sets of different sizes and characteristics. For gaining insights from the data, we show that our solution is more efficient and effective than simply asking users to extract relationships via standard interaction techniques, especially when the data set is large and the relationships are complex. We also collect expert feedback to confirm the usefulness of our approach.'
# }
# paper_scraper.train(url, data)
#
# # Loop through clusters, aggregate information.
# with open('vis_papers.csv', 'a') as output_file:
#     # Initialize files.
#      dict_writer = csv.DictWriter(
#          output_file,
#          ["cluster_title", "conference", "year", "title", "abstract", "keywords", "url"]
#      )
#     dict_writer.writeheader()
#     error_file = open('vis_paper_errors.txt', 'a')
#
#     # Note: Stopped last time with cluster Cameras, Camera Views and Projections (no papers from this cluster yet).
#     for i, cluster_title in enumerate(parsed_content_list):
#         if i >= 176 and cluster_title != "":
#             # Remove trailing and leading whitespaces.
#             clean_cluster_title = cluster_title.strip()
#
#             # Print progress.
#             print(str(i) + " | " + str(i / float(len(parsed_content_list)) * 100) + "% | " + clean_cluster_title)
#
#             # Get detailled paper information.
#             response = urlopen(
#                 Request(
#                     'http://keyvis.org/load3.php',
#                     urlencode({'expert': str(i - 1)}).encode()
#                 )
#             ).read().decode()
#             parsed_paper_response = h.handle(response)
#
#             # Dismiss keywords at beginning, only keep from first paper on.
#             parsed_paper_response = parsed_paper_response[
#                 parsed_paper_response.index("---|---|---") + 11:
#             ]
#
#             # Split by ---|---|--- to get information on individual papers.
#             parsed_paper_list = parsed_paper_response.strip().split("---|---")
#
#             # Loop through papers, collect information on papers.
#             for paper in parsed_paper_list:
#                 paper_info = {
#                     "cluster_title": clean_cluster_title,
#                     "conference": None,
#                     "year": None,
#                     "title": None,
#                     "abstract": None,
#                     "keywords": None,
#                     "url": None
#                 }
#
#                 # Split paper in description to extract conference, year and link.
#                 paper_parts = paper.split("|")
#
#                 try:
#                     if len(paper_parts) > 1:
#                         paper_info["conference"] = paper_parts[0].strip()
#                         paper_info["year"] = paper_parts[1].strip()
#                         # Get keywords - remove everything up to first occurence of [, then start replacing abstract
#                         # with normalized separators.
#                         keywords_raw = paper_parts[len(paper_parts) - 1]
#                         paper_info["keywords"] = keywords_raw[keywords_raw.index('[') + 1:] \
#                             .replace('[ ', ',') \
#                             .replace('[', ',') \
#                             .replace('\n', '') \
#                             .replace('  ', '') \
#                             .strip()
#
#                         # For URL to paper: Find last occurence of round bracket, read URL except closing round
#                         # bracket).
#                         paper_info["url"] = paper_parts[2][paper_parts[2].rfind('(') + 1:-2]
#                         content = paper_scraper.scrape(paper_info["url"])[0]
#                         # Get paper title.
#                         paper_info["title"] = h.handle(content["title"][0]) \
#                             .replace('\n', ' ') \
#                             .replace('\r', '') \
#                             .replace(" - IEEE Journals & Magazine", "") \
#                             .strip()
#                         # Get abstract.
#                         abstract = h.handle(content["abstract"][0]) \
#                             .replace('\n', ' ') \
#                             .replace('\r', '') \
#                             .strip()
#                         abstract_index = abstract.index('"abstract":') + 12
#                         abstract_end_index = abstract_index + (abstract[abstract_index + 1:].index('","')) + 1
#                         paper_info["abstract"] = abstract[abstract_index:abstract_end_index]
#
#                     if paper_info["abstract"] is not None:
#                         # Dump paper info list to CSV.
#                         dict_writer.writerow(paper_info)
#                         print("    " + str(paper_info))
#
#                 except Exception as e:
#                     print("*************")
#                     print("Error: " + paper_info["url"])
#                     error_file.write(paper_info["url"] + "\n")
#                     print("*************")
#
#     error_file.close()

# Open error file, try with alternative scraper.
paper_scraper_alt = scrapely.Scraper()
url = "https://www.computer.org/csdl/proceedings/visual/2000/6478/00/00885702-abs.html"
data = {
    'title': 'Image based rendering with stable frame rates',
    'abstract': 'Presents an efficient keyframeless image-based rendering technique. An intermediate image is used to exploit the coherences among neighboring frames. The pixels in the intermediate image are first rendered by a ray-casting method and then warped to the intermediate image at the current viewpoint and view direction. We use an offset buffer to record the precise positions of these pixels in the intermediate image. Every frame is generated in three steps: warping the intermediate image onto the frame, filling in holes, and selectively rendering a group of "old" pixels. By dynamically adjusting the number of those "old" pixels in the last step, the workload at every frame can be balanced. The pixels generated by the last two steps make contributions to the new intermediate image. Unlike occasional keyframes in conventional image-based rendering, which need to be totally re-rendered, intermediate images only need to be partially updated at every frame. In this way, we guarantee more stable frame rates and more uniform image qualities. The intermediate image can be warped efficiently by a modified incremental 3D warp algorithm. As a specific application, we demonstrate our technique with a voxel-based terrain rendering system.',
    'keywords': 'rendering (computer graphics), ray tracing, image morphing, real-time systems',
    'year': 'Proceedings Visualization 2000. VIS 2000 (2000)'
}
paper_scraper_alt.train(url, data)

# Build maps for reverse lookup.
url_to_cluster = {}
url_to_conference = {}
for i, cluster_title in enumerate(parsed_content_list):
    # Remove trailing and leading whitespaces.
    clean_cluster_title = cluster_title.strip()
    print(clean_cluster_title)

    # Get detailled paper information.
    response = urlopen(
        Request(
            'http://keyvis.org/load3.php',
            urlencode({'expert': str(i - 1)}).encode()
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
        # Split paper in description to extract conference, year and link.
        paper_parts = paper.split("|")

        if len(paper_parts) > 1:
            # Extract URL.
            url = paper_parts[2][paper_parts[2].rfind('(') + 1:-2]
            # Update reverse lookup dictionaries.
            url_to_cluster[url] = clean_cluster_title
            url_to_conference[url] = paper_parts[0].strip()

# Loop through entries in error file, try with alternative scraper.
with open("vis_paper_errors.txt") as error_file:
    # Make sure we don't process URLs twice (in case our error file is messed up).
    processed_urls_set = set()

    with open('vis_papers.csv', 'a') as output_file:
        # Initialize files.
        dict_writer = csv.DictWriter(
            output_file,
            ["cluster_title", "conference", "year", "title", "abstract", "keywords", "url"]
        )

        for url in [line.rstrip('\n') for line in error_file]:
            if url not in processed_urls_set:
                print(url)
                processed_urls_set.add(url)
                try:
                    content = paper_scraper_alt.scrape(url)[0]
                    year_index = content["year"][0].index('(') + 1

                    dict_writer.writerow({
                        "cluster_title": url_to_cluster[url],
                        "conference": url_to_conference[url],
                        "year": content["year"][0][year_index:year_index + 4],
                        "title": content["title"][0],
                        "abstract": content["abstract"][0],
                        "keywords": content["keywords"][0],
                        "url": url
                    })

                except urllib.error.HTTPError as e:
                    print("URL " + url + " does not exist. Skipping.")

                except Exception as e:
                    print("Other error with URL " + url)

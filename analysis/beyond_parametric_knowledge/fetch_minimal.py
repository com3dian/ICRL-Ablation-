import urllib.request
import xml.etree.ElementTree as ET
import json
import time
from tqdm import tqdm

def fetch_cs_papers(num_papers=150):
    papers = []
    batch_size = 20  # Fetch in smaller batches
    
    for start in tqdm(range(0, num_papers, batch_size)):
        current_batch = min(batch_size, num_papers - start)
        
        # Query for CS papers
        url = f'http://export.arxiv.org/api/query?search_query=cat:cs.*&start={start}&max_results={current_batch}&sortBy=submittedDate&sortOrder=descending'
        
        print(f"Fetching papers {start+1} to {start+current_batch}...")
        
        data = urllib.request.urlopen(url)
        response = data.read().decode('utf-8')
        
        # Parse XML
        root = ET.fromstring(response)
        
        # Namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            paper = {}
            
            # Get title
            title = entry.find('atom:title', ns)
            paper['title'] = title.text.strip() if title is not None else ''
            
            # Get abstract
            abstract = entry.find('atom:summary', ns)
            paper['abstract'] = abstract.text.strip() if abstract is not None else ''
            
            # Get arxiv ID
            id_elem = entry.find('atom:id', ns)
            if id_elem is not None:
                paper['arxiv_id'] = id_elem.text.strip().split('/')[-1]
            # Get published date
            published = entry.find('atom:published', ns)
            if published is not None:
                paper['published'] = published.text.strip()

            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text.strip())
            paper['authors'] = authors
            
            papers.append(paper)

        # To-do: continue to store to JSON
        with open('arxiv_papers_1500.json', 'w') as f:
            json.dump(papers, f, indent=2)
        
        # Be polite to the API
        if start + batch_size < num_papers:
            time.sleep(3)
    
    return papers

# Fetch 150 papers
papers = fetch_cs_papers(1500)

# Save to JSON
with open('arxiv_papers_150.json', 'w') as f:
    json.dump(papers, f, indent=2)

print(f"\nTotal papers fetched: {len(papers)}")
print(f"Saved to: arxiv_papers_150.json")

# Display first 3 papers
for i, paper in enumerate(papers[:3], 1):
    print(f"\nPaper {i}:")
    print(f"Title: {paper['title'][:80]}...")
    print(f"Authors: {', '.join(paper['authors'][:2])}...")
    print(f"Abstract: {paper['abstract'][:150]}...")
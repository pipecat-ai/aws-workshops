import os
import json
import requests
from bs4 import BeautifulSoup
from googlesearch import search

def search_health_info(query):
    """
    Search for health information using Google Search
    """
    # Add "health" to the query to focus on health-related results
    search_query = f"{query} health information"
    
    # Perform the search - get top 5 results
    search_results = []
    try:
        for url in search(search_query, num_results=5):
            search_results.append(url)
    except Exception as e:
        print(f"Search error: {e}")
        return []
    
    # Format the results
    formatted_results = []
    for url in search_results:
        try:
            # Get the page content
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Parse the HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else url
                
                # Extract a snippet (first paragraph or meta description)
                snippet = ""
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    snippet = meta_desc.get('content')
                else:
                    first_p = soup.find('p')
                    if first_p:
                        snippet = first_p.get_text()[:200] + "..."
                
                # Extract source domain
                source = url.split('//')[1].split('/')[0]
                
                formatted_results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": url,
                    "source": source
                })
        except Exception as e:
            print(f"Error processing {url}: {e}")
    
    return formatted_results

def summarize_search_results(results, query):
    """
    Create a summary of search results that can be used by the agent
    """
    if not results:
        return f"I couldn't find any information about '{query}'."
    
    summary = f"Here's what I found about '{query}':\n\n"
    
    for i, result in enumerate(results[:3], 1):  # Limit to top 3 results
        summary += f"{i}. {result['title']}\n"
        summary += f"   {result['snippet']}\n"
        summary += f"   Source: {result['source']}\n\n"
    
    summary += "This information is from web searches and should not replace professional medical advice."
    
    return summary

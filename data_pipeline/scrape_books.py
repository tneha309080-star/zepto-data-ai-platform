import re
import time
from pathlib import Path
from turtle import title

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://books.toscrape.com/"
PAGES_TO_SCRAPE = 5
GBP_TO_INR = 105.50

OUTPUT_DIR = Path(__file__).resolve().parent
RAW_OUTPUT = OUTPUT_DIR / "books_raw.csv"
CLEAN_OUTPUT = OUTPUT_DIR / "books_clean.csv"


def parse_rating(rating_text):
    """Convert star-rating text such as 'Three' into an integer 1-5."""
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }

    return rating_map.get(rating_text)


def parse_price(price_text):
    """Convert a GBP price such as '£51.77' into a float."""
    match = re.search(r"[\d.]+", price_text)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def scrape_page(page_number):
    """Scrape one catalog page."""
    if page_number == 1:
        url = BASE_URL + "index.html"
    else:
        url = BASE_URL + f"catalogue/page-{page_number}.html"

    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.select("article.product_pod")

    rows = []

    for book in books:
        try:
            title_tag = book.select_one("h3 a")
            price_tag = book.select_one("p.price_color")
            rating_tag = book.select_one("p.star-rating")
            availability_tag = book.select_one("p.instock.availability")

            if any(tag is None for tag in (title_tag, price_tag, rating_tag, availability_tag)):
                continue

            title = title_tag.get("title", "").strip()
            price_gbp = parse_price(price_tag.get_text(strip=True))

            rating_text = next(
                (item for item in rating_tag.get("class", [])
                 if item in ["One", "Two", "Three", "Four", "Five"]),
                None,
            )
            rating = parse_rating(rating_text)

            availability = availability_tag.get_text(" ", strip=True)
            in_stock = "In stock" in availability

            category = "Unknown"

            detail_link = title_tag.get("href")
            if detail_link:
                detail_url = urljoin(url, detail_link)

                try:
                    detail_response = requests.get(
                        detail_url,
                        timeout=20,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    detail_response.raise_for_status()
                    detail_soup = BeautifulSoup(
                        detail_response.text, "html.parser"
                    )
                    breadcrumb_links = detail_soup.select(
                        "ul.breadcrumb li a"
                    )
                    if len(breadcrumb_links) >= 3:
                        category = breadcrumb_links[-1].get_text(strip=True)
                except requests.RequestException as exc:
                    print(f"Could not get category for {title}: {exc}")

            rows.append(
                {
                    "title": title,
                    "price_gbp": price_gbp,
                    "rating": rating,
                    "availability": availability,
                    "in_stock": in_stock,
                    "category": category,
                }
            )

        except Exception as exc:
            print(f"Skipping book because of parsing error: {exc}")

    return rows


def clean_data(df):
    """Clean scraped data and create INR price."""
    df = df.copy()

    df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["in_stock"] = df["in_stock"].astype(bool)

    # Assignment allows numeric median imputation for parse failures.
    if df["price_gbp"].isna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    if df["rating"].isna().any():
        df["rating"] = df["rating"].fillna(df["rating"].median())

    df["rating"] = df["rating"].round().astype(int)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df


def main():
    all_rows = []

    print(f"Scraping first {PAGES_TO_SCRAPE} catalog pages...")

    for page_number in range(1, PAGES_TO_SCRAPE + 1):
        print(f"Scraping page {page_number}/{PAGES_TO_SCRAPE}...")

        try:
            page_rows = scrape_page(page_number)
            all_rows.extend(page_rows)
            print(f"  Collected {len(page_rows)} books.")
        except requests.RequestException as exc:
            print(f"  Page {page_number} failed: {exc}")

        time.sleep(0.5)

    raw_df = pd.DataFrame(all_rows)

    if raw_df.empty:
        raise RuntimeError("No books were scraped.")

    raw_df.to_csv(RAW_OUTPUT, index=False)

    clean_df = clean_data(raw_df)

    clean_df.to_csv(CLEAN_OUTPUT, index=False)

    print()
    print("Scraping complete.")
    print(f"Total books: {len(clean_df)}")
    print(f"Unique categories: {clean_df['category'].nunique()}")
    print()
    print("Category distribution:")
    print(clean_df["category"].value_counts())
    print()
    print(f"Raw output: {RAW_OUTPUT}")
    print(f"Clean output: {CLEAN_OUTPUT}")


if __name__ == "__main__":
    main()
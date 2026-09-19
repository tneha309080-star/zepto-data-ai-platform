import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "books_clean.csv"
DB_FILE = BASE_DIR / "books.db"
OUTPUT_FILE = BASE_DIR / "sql_outputs.txt"

GBP_TO_INR = 105.50


def create_database():
    # ---------------------------------------------------------
    # 1. Load cleaned data
    # ---------------------------------------------------------
    df = pd.read_csv(CSV_FILE)

    # ---------------------------------------------------------
    # 2. Verify / enforce GBP -> INR conversion
    # ---------------------------------------------------------
    df["price_inr_expected"] = df["price_gbp"] * GBP_TO_INR

    conversion_matches = (
        (df["price_inr"] - df["price_inr_expected"]).abs() < 0.01
    ).all()

    print("\n--- Conversion Verification ---")
    print(f"Conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR")
    print(f"105.50 GBP = {105.50 * GBP_TO_INR:.2f} INR")
    print(f"All CSV conversions correct: {conversion_matches}")

    # Remove temporary verification column
    df.drop(columns=["price_inr_expected"], inplace=True)

    # ---------------------------------------------------------
    # 3. Create SQLite database
    # ---------------------------------------------------------
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    # Enable foreign-key enforcement in SQLite
    cursor.execute("PRAGMA foreign_keys = ON")

    # ---------------------------------------------------------
    # 4. Create normalized category table
    # ---------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # ---------------------------------------------------------
    # 5. Create normalized books table
    # ---------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            availability TEXT,
            in_stock BOOLEAN,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Clear old data so script can be safely re-run
    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    # ---------------------------------------------------------
    # 6. Insert categories
    # ---------------------------------------------------------
    categories = sorted(df["category"].dropna().unique())

    for category in categories:
        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,),
        )

    # Create category lookup
    category_lookup = {
        row[1]: row[0]
        for row in cursor.execute(
            """
            SELECT category_id, category_name
            FROM categories
            """
        )
    }

    # ---------------------------------------------------------
    # 7. Insert books
    # ---------------------------------------------------------
    for _, row in df.iterrows():

        category_id = category_lookup[row["category"]]

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                availability,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                row["availability"],
                bool(row["in_stock"]),
                category_id,
            ),
        )

    connection.commit()

    # ---------------------------------------------------------
    # 8. Verify PK / FK schema
    # ---------------------------------------------------------
    print("\n--- Schema Verification ---")

    categories_schema = cursor.execute(
        "PRAGMA table_info(categories)"
    ).fetchall()

    books_schema = cursor.execute(
        "PRAGMA table_info(books)"
    ).fetchall()

    foreign_keys = cursor.execute(
        "PRAGMA foreign_key_list(books)"
    ).fetchall()

    print("Categories table:")
    for column in categories_schema:
        print(column)

    print("\nBooks table:")
    for column in books_schema:
        print(column)

    print("\nBooks foreign keys:")
    for fk in foreign_keys:
        print(fk)

    # ---------------------------------------------------------
    # 9. Required SQL queries
    # ---------------------------------------------------------
    queries = {

        # Query 1: SELECT + WHERE
        "Q1_SELECT_WHERE": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4
        """,

        # Query 2: ORDER BY + LIMIT
        "Q2_ORDER_BY_LIMIT": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10
        """,

        # Query 3: DISTINCT
        "Q3_DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating
        """,

        # Query 4: IN
        "Q4_IN": """
            SELECT title, rating, category_id
            FROM books
            WHERE rating IN (4, 5)
        """,

        # Query 5: BETWEEN
        "Q5_BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30
            ORDER BY price_gbp
        """,

        # Query 6: JOIN
        "Q6_JOIN": """
            SELECT
                b.book_id,
                b.title,
                b.price_gbp,
                b.price_inr,
                b.rating,
                c.category_name
            FROM books AS b
            JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY c.category_name, b.title
        """,

        # Query 7: JOIN + aggregation
        "Q7_CATEGORY_SUMMARY": """
            SELECT
                c.category_name,
                COUNT(b.book_id) AS book_count,
                ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
            FROM categories AS c
            JOIN books AS b
                ON c.category_id = b.category_id
            GROUP BY c.category_id, c.category_name
            ORDER BY book_count DESC
        """,
    }

    # ---------------------------------------------------------
    # 10. Run SQL queries and save outputs
    # ---------------------------------------------------------
    print("\n--- SQL Query Results ---")

    output_lines = []

    for query_name, query in queries.items():

        print(f"\n{query_name}")
        print("-" * 60)
        print(query.strip())

        result = cursor.execute(query).fetchall()

        for row in result[:10]:
            print(row)

        if len(result) > 10:
            print(f"... showing first 10 of {len(result)} rows")

        output_lines.append(f"\n{query_name}")
        output_lines.append("=" * 60)
        output_lines.append(query.strip())
        output_lines.append("\nOutput:")

        for row in result:
            output_lines.append(str(row))

    # ---------------------------------------------------------
    # 11. pandas read_sql()
    # ---------------------------------------------------------
    print("\n--- pandas.read_sql() ---")

    df_sql_1 = pd.read_sql(
        queries["Q1_SELECT_WHERE"],
        connection
    )

    df_sql_2 = pd.read_sql(
        queries["Q6_JOIN"],
        connection
    )

    print("\nResult from Q1 using pd.read_sql():")
    print(df_sql_1.head())

    print("\nResult from Q6 using pd.read_sql():")
    print(df_sql_2.head())

    # ---------------------------------------------------------
    # 12. Reproduce JOIN using pd.merge()
    # ---------------------------------------------------------
    print("\n--- pandas.merge() JOIN Verification ---")

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            category_id
        FROM books
        """,
        connection,
    )

    categories_df = pd.read_sql(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection,
    )

    merged_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
    )

    merged_df = merged_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "category_name",
        ]
    ].sort_values(["category_name", "title"])

    sql_join_df = df_sql_2.sort_values(
        ["category_name", "title"]
    ).reset_index(drop=True)

    merged_df = merged_df.reset_index(drop=True)

    # Check that SQL JOIN and pandas.merge() produce equivalent results
    join_columns = [
        "book_id",
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "category_name",
    ]

    join_matches = merged_df[join_columns].equals(
        sql_join_df[join_columns]
    )

    print("SQL JOIN result:")
    print(sql_join_df.head())

    print("\npandas.merge() result:")
    print(merged_df.head())

    print(f"\nSQL JOIN == pandas.merge(): {join_matches}")

    output_lines.append("\n\nPANDAS read_sql() / merge() VERIFICATION")
    output_lines.append("=" * 60)
    output_lines.append(
        f"SQL JOIN rows: {len(sql_join_df)}"
    )
    output_lines.append(
        f"pandas.merge() rows: {len(merged_df)}"
    )
    output_lines.append(
        f"SQL JOIN == pandas.merge(): {join_matches}"
    )

    # ---------------------------------------------------------
    # 13. Verify row counts
    # ---------------------------------------------------------
    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    print("\n--- Final Database Summary ---")
    print(f"Books inserted: {book_count}")
    print(f"Categories inserted: {category_count}")

    output_lines.append("\n\nDATABASE SUMMARY")
    output_lines.append("=" * 60)
    output_lines.append(f"Books inserted: {book_count}")
    output_lines.append(f"Categories inserted: {category_count}")
    output_lines.append(
        f"Conversion rate: 1 GBP = {GBP_TO_INR:.2f} INR"
    )
    output_lines.append(
        f"105.50 GBP = {105.50 * GBP_TO_INR:.2f} INR"
    )
    output_lines.append(
        f"All CSV conversions correct: {conversion_matches}"
    )
    output_lines.append(
        f"SQL JOIN == pandas.merge(): {join_matches}"
    )

    # Save query strings and important verification outputs
    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    connection.close()

    print(f"\nDatabase created: {DB_FILE}")
    print(f"SQL outputs saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_database()

import os
import psycopg2
import requests
from datetime import date
from dotenv import load_dotenv
from lxml import html


def is_job_active(url: str) -> bool:
    """
    Returns True if vacancy is active, False if closed.
    """
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        return False  # 404 or redirect means closed
    tree = html.fromstring(response.content)

    # Work.ua shows specific messages on closed pages
    closed_markers = ["На жаль, вакансію не знайдено"]

    deleted_text = tree.xpath("//h1[@class='mt-0 h3']/text()")
    for marker in closed_markers:
        if marker in deleted_text:
            return False

    return True


def check_vacancies():
    load_dotenv()
    conn = psycopg2.connect(
            host=f"{os.getenv('DB_HOST')}",
            database=f"{os.getenv('DB_NAME')}",
            user=f"{os.getenv('DB_USER')}",
            password=f"{os.getenv('DB_PASSWORD')}",
            port=f"{os.getenv('DB_PORT')}")
    cur = conn.cursor()

    # get all active vacancies
    cur.execute("""
        SELECT job_id, url
        FROM jobs
        WHERE is_deleted = FALSE;
    """)
    vacancies = cur.fetchall()
    print(f"Found {len(vacancies)} vacancies to check")

    for job_id, url in vacancies:
        print(f"Checking {job_id} -> {url}")

        active = is_job_active(url)

        if not active:
            print(f"❌ Vacancy CLOSED: {job_id}")

            cur.execute("""
                    SELECT posted_date
                    FROM jobs
                    WHERE job_id = %s;
                """, (job_id,))

            posted_date = cur.fetchone()[0]
            print(f"Posted date: {posted_date}")
            closed_date = date.today()
            time_to_hire = (closed_date - posted_date).days
            print(f"Time to hire: {time_to_hire}")

            cur.execute("""
                    UPDATE jobs
                    SET is_deleted = TRUE,
                        closed_date = %s,
                        time_to_hire = %s
                    WHERE job_id = %s;
                """, (closed_date, time_to_hire, job_id))
            conn.commit()

        else:
            print(f"✅ Vacancy ACTIVE: {job_id}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    check_vacancies()

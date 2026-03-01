from playwright.sync_api import sync_playwright

def verify_generation_page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Go to the Generation page
        page.goto("http://localhost:5173/LightGNN-Peptide/generation")

        # Wait for the page to load
        page.wait_for_selector("h1:has-text('Peptide Generation')")

        # Fill in parameters
        page.fill("input[placeholder='e.g., PDB:1A2B']", "PDB:1TEST")
        page.fill("input[placeholder='e.g., 10']", "15")

        # Click Generate button
        page.click("button:has-text('Generate Peptides')")

        # Wait for results to appear (the mock backend returns quickly, but we wait for the card)
        # The result card has "Stability" text
        try:
            page.wait_for_selector("text=Stability", timeout=5000)
            print("Results appeared successfully")
        except:
            print("Timed out waiting for results")

        # Take screenshot
        page.screenshot(path="frontend_verification/generation_test.png")

        browser.close()

if __name__ == "__main__":
    verify_generation_page()


## `website.py` hosted entry point

To host the Sightengine-based alternate UI, choose `website.py` as the entry point
in Streamlit Community Cloud. Configure `SIGHTENGINE_API_USER` and
`SIGHTENGINE_API_SECRET` under the app's **Settings → Secrets**. For local use,
store them in `.streamlit/secrets.toml`; never commit that file. Rotate any keys
that were previously hard-coded in source. This version sends uploaded images to
Sightengine for analysis.

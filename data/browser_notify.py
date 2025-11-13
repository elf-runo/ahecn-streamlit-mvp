from streamlit.components.v1 import html
import json

def browser_notify(title, body):
    html(f"""
    <script>
    (async () => {{
      const perm = await Notification.requestPermission();
      if (perm === "granted") {{
        new Notification({json.dumps(title)}, {{body:{json.dumps(body)}}});
      }}
    }})();
    </script>
    """, height=0)

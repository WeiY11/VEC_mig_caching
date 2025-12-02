
print("Starting import...")
try:
    import train_single_agent
    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
except SystemExit:
    print("SystemExit during import.")

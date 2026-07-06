from utils import tool_belt as tb


def main(slider_num: int, slot_num: int):
    """
    Moves a specific filter slider to a designated slot.
    Includes offline detection, abort handling, and prevents hanging.
    """
    print(f"Initiating movement for slider {slider_num} to slot {slot_num}...")

    # 1. Validate inputs
    if not (1 <= slider_num <= 3):
        raise ValueError(f"Invalid filter: {slider_num}. Must be 1, 2, or 3.")
    if not (0 <= slot_num <= 4):
        raise ValueError(f"Invalid slot: {slot_num}. Must be between 0 and 4.")

    # 2. Connect to the correct server (WITH HARD KILL SWITCHES)
    slider_server = None
    try:
        if slider_num == 1:
            slider_server = tb.get_server_slider_1()
        elif slider_num == 2:
            slider_server = tb.get_server_slider_2()
        elif slider_num == 3:
            slider_server = tb.get_server_slider_3()
    except Exception as e:
        # HARD FAIL: Stops the laser script entirely
        raise RuntimeError(
            f"Slider {slider_num} server is OFFLINE or unreachable. Details: {e}"
        )

    # Some APIs return None instead of throwing an error when offline
    if slider_server is None:
        # HARD FAIL
        raise RuntimeError(
            f"Slider {slider_num} server is OFFLINE (Connection returned None)."
        )

    # 3. Check current position to prevent hanging
    try:
        current_slot = slider_server.get_filter()
        if current_slot == slot_num:
            print(
                f"Slider {slider_num} is already at slot {slot_num}. Skipping movement."
            )
            return
    except AttributeError:
        pass  # Ignored if the command doesn't exist
    except Exception as e:
        print(
            f"[!] Warning: Could not verify current position of Slider {slider_num}: {e}"
        )

    # 4. Execute the movement with Abort capability
    try:
        slider_server.set_filter(slot_num)
        print(f"Successfully moved slider {slider_num} to slot {slot_num}.")

    except KeyboardInterrupt:
        # Fixed the undefined 'e' variable here
        raise RuntimeError(
            f"Sequence aborted by user while moving slider {slider_num}."
        )

    except Exception as e:
        # HARD FAIL: Instantly kills the script if the slider gets stuck physically
        raise RuntimeError(f"Hardware failure while moving slider {slider_num}: {e}")


if __name__ == "__main__":
    # If run directly, you'll need to pass arguments now since defaults were removed
    main(1, 0)

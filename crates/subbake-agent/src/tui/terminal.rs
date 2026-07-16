use std::io;

use crossterm::ExecutableCommand;
use crossterm::event::{
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
    supports_keyboard_enhancement,
};

pub(super) struct TerminalSessionGuard {
    active: bool,
    keyboard_enhancement: bool,
    alternate_screen: bool,
}

impl TerminalSessionGuard {
    pub(super) fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        let keyboard_enhancement = supports_keyboard_enhancement().unwrap_or(false);
        if keyboard_enhancement
            && let Err(error) = io::stdout().execute(PushKeyboardEnhancementFlags(
                KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
            ))
        {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self {
            active: true,
            keyboard_enhancement,
            alternate_screen: false,
        })
    }

    pub(super) fn enter_alternate_screen(&mut self) -> io::Result<()> {
        if !self.alternate_screen {
            io::stdout().execute(EnterAlternateScreen)?;
            self.alternate_screen = true;
        }
        Ok(())
    }

    pub(super) fn leave_alternate_screen(&mut self) -> io::Result<()> {
        if self.alternate_screen {
            io::stdout().execute(LeaveAlternateScreen)?;
            self.alternate_screen = false;
        }
        Ok(())
    }

    pub(super) fn restore(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        self.active = false;
        let screen_result = self.leave_alternate_screen();
        let keyboard_result = if self.keyboard_enhancement {
            io::stdout()
                .execute(PopKeyboardEnhancementFlags)
                .map(|_| ())
        } else {
            Ok(())
        };
        let raw_result = disable_raw_mode();
        screen_result.and(keyboard_result).and(raw_result)
    }
}

impl Drop for TerminalSessionGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

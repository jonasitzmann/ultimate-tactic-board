Known Issues
---

- players always rotate clockwise

- disc recognition unreliable
- players are not recognized correctly if they appear to cross the boundary
- player recognition / identification fails if the player's frame is not clearly separated from its number
- the largest white rectangle is detected as the field with no further checks
- linear interpolation of  player's poses results in unrealistic movements
  - estimate player speeds
  - define max acceleration
  - only allow sideway motion at low speed
  - at high speed: turn -> move -> turn
#let styled-table(..args) = {
  table(
    ..args,
    stroke: (col, row) => {
      if row <= 0 {
        (bottom: 1pt + black)
      }

      if row > 0 and col != 0 {
        (left: 0.125pt)
      }
    },
    inset: 3pt,
    align: center + horizon,
    fill: (col, row) => {
      if row == 0 { gray.lighten(60%) } else {
        if calc.even(row) { gray.lighten(80%) } else { none }
      }
    }
  )
}


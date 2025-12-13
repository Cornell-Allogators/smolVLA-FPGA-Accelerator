#let styled-table(..args) = {
  table(
    ..args,
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: center + horizon,
    fill: (col, row) => {
      if row == 0 { gray.lighten(40%) } 
      else { none }
    }
  )
}


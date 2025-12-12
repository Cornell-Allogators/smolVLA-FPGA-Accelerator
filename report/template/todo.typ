#let todo(
  who,
  body,
  done: false,
) = {
  set align(center)
  rect(
    fill: if done {
      green.lighten(80%)
    } else {
      red.lighten(80%)
    },
    inset: 8pt,
    radius: 8pt,
    stroke: 1pt,
    width: 80%,
    if done {
      align(left)[
        #text(size: 14pt)[
          *Written By: *#who
        ]
      ]
    } else {
      align(left)[
        #text(size: 14pt)[
          *TODO: *#who
        ] \ #text[
          #body
        ]
      ]
    },
  )
}

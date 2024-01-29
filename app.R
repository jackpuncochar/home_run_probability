# R Shiny application that displays results of logistic regression HR probability model

# Load packages
library(shiny)

# UI
ui <- fluidPage(
  titlePanel("Scoreboard"),
  fluidRow(
    column(4,
      h1("Outs"),
      h3(uiOutput("outs"))
    ),
    column(4,
      h1("Balls"),
      h3(uiOutput("balls"))
    ),
    column(4,
      h1("Strikes"),
      h3(uiOutput("strikes"))
    )
  ),
  br(),
  fluidRow(
    column(4,
           h1("Pitching"),
           h3(uiOutput("pitching_info"))
    ),
    column(4,
           h1("At-Bat"),
           h3(uiOutput("hitting_info"))
    ),
    column(4,
           h1("HR Probability"),
           h3(uiOutput("hr_prob"))
    )
  ),
  br(),
  fluidRow(
    column(4,
      h1("Inning:")
    ),
    column(4,
      h1(uiOutput("away_score"))
    ),
    column(4,
      h1(uiOutput("home_score"))
    )
  ),
  fluidRow(
    column(4,
      h1(uiOutput("inning"))
    ),
    column(4,
      imageOutput("Away_Team_Image")
    ),
    column(4,
      imageOutput("Home_Team_Image")      
    )
  )
)

# Server
server <- function(input, output, session) {
  output$inning <- renderText({
    live_pitch$about.inning
  })
  output$away_score <- renderText({
    live_pitch$result.awayScore
  })
  output$home_score <- renderText({
    live_pitch$result.homeScore
  })
  output$pitching_info <- renderText({
    paste0(live_pitch$matchup.pitcher.fullName, " (", live_pitch$p_throws, ")")
  })
  output$hitting_info <- renderText({
    paste0(live_pitch$matchup.batter.fullName, " (", live_pitch$stand, ")")
  })
  output$hr_prob <- renderText({
    paste(round(live_pitch$new_pred, 3))
  })
  output$Home_Team_Image <- renderImage({
    filename = normalizePath(file.path("./www/", paste(paste0("mlb-", live_pitch$home_team, "-logo", sep=""), ".png", sep="")))
    return(list(src = filename, alt=live_pitch$home_team, width=100, height=100))
  }, deleteFile = FALSE)
  output$Away_Team_Image <- renderImage({
    filename = normalizePath(file.path("./www/", paste(paste0("mlb-", live_pitch$away_team, "-logo", sep=""), ".png", sep="")))
    return(list(src = filename, alt=live_pitch$away_team, width=100, height=100))
  }, deleteFile = FALSE)
  output$balls <- renderText({
    sub("_.*", "", live_pitch$count)
  })
  output$strikes <- renderText({
    sub(".*_", "", live_pitch$count)
  })
  output$outs <- renderText({
    live_pitch$outs_when_up
  })
}

shinyApp(ui, server)


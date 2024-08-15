/*
 *                      *** Used by CI ***
 * This file is used by deploy-pr.yml when building the PR Previews for QML.
 * This is a modified gatsby-node.ts file that replaces the gatsby-node in pennylane-website build
 * and remove the dependency on other SWC backend service that the QML PR Previews does not need.
 */

/* eslint-disable @typescript-eslint/no-var-requires */
const { createFilePath } = require(`gatsby-source-filesystem`)
import {
  CreateBabelConfigArgs,
  CreateNodeArgs,
  GatsbyNode,
} from 'gatsby'
import path from 'path'

/**
 * onCreateNode - Called when a new node is created. Plugins wishing to extend or transform nodes created by other plugins should implement this API.
 * https://www.gatsbyjs.com/docs/reference/config-files/gatsby-node/#onCreateNode
 * @type {import('gatsby').GatsbyNode['onCreateNode']}
 */
exports.onCreateNode = ({ node, actions, getNode }: CreateNodeArgs) => {
  const { createNodeField } = actions

  if (node.internal.type === `MarkdownRemark`) {
    const value = createFilePath({ node, getNode })

    createNodeField({
      name: `slug`,
      node,
      value,
    })
  }
}

/**
 * onCreateBabelConfig - Let plugins extend/mutate the site’s Babel configuration by calling setBabelPlugin or setBabelPreset.
 * https://www.gatsbyjs.com/docs/reference/config-files/gatsby-node/#onCreateBabelConfig
 */
exports.onCreateBabelConfig = ({ actions }: CreateBabelConfigArgs) => {
  actions.setBabelPlugin({
    name: '@babel/plugin-transform-react-jsx',
    options: {
      runtime: 'automatic',
    },
  })
}

/**
 * createSchemaCustomization - Customize Gatsby’s GraphQL schema by creating type definitions, field extensions or adding third-party schemas.
 * https://www.gatsbyjs.com/docs/reference/config-files/gatsby-node/#createSchemaCustomization
 * @type {import('gatsby').GatsbyNode['createSchemaCustomization']}
 */
exports.createSchemaCustomization = ({
  actions,
}: {
  actions: {
    createTypes: (types: string) => void
  }
}) => {
  const { createTypes } = actions

  // Explicitly define the siteMetadata {} object
  // This way those will always be defined even if removed from gatsby-config.js

  // Also explicitly define the Markdown frontmatter
  // This way the "MarkdownRemark" queries will return `null` even when no
  // blog posts are stored inside "content/blog" instead of returning an error
  createTypes(`
    type SiteSiteMetadata {
      author: Author
      siteUrl: String
      social: Social
    }

    type Author {
      name: String
      summary: String
    }

    type Social {
      twitter: String
    }

    type MarkdownRemark implements Node {
      frontmatter: Frontmatter
      fields: Fields
    }

    type Frontmatter {
      title: String
      description: String
      date: Date @dateformat
      categories: [String!]
      relatedContent: [RelatedContent]
      hardware: [Hardware]
    }

    type Fields {
      slug: String
    }

    type RelatedContent {
      id: String!
      title: String!
      previewImages: [PreviewImages]
    }

    type PreviewImages {
      type: String!
      uri: String!
    }

    type Hardware {
      id: String!
      link: String!
      logo: String!
    }
  `)
}

/**
 * Import UI templates
 * Templates are used in createPages to programmatically create page.
 */

const demoPage = path.resolve(
  `${__dirname}/src/templates/demos/individualDemo/demo.tsx`
)

/**
 * createPages - Create pages dynamically.
 * This extension point is called only after the initial sourcing and transformation of nodes
 * plus creation of the GraphQL schema are complete so you can query your data in order to create pages.
 * https://www.gatsbyjs.com/docs/reference/config-files/gatsby-node/#onCreateWebpackConfig
 */
export const createPages: GatsbyNode['createPages'] = async ({
  graphql,
  actions,
  reporter,
}) => {
  const { createPage } = actions

  // Queries
  // -------

  // Query .md files from /content/demos directory
  const DemosResults = await graphql<Queries.GetDemoDataQuery>(`
    query GetDemoData {
      allMarkdownRemark(filter: { fileAbsolutePath: { regex: "/demos/" } }) {
        nodes {
          frontmatter {
            slug
            title
            meta_description
          }
          id
        }
      }
    }
  `)

  if (DemosResults.errors) {
    reporter.panicOnBuild(
      '🚨  ERROR: Loading "createPages" query for demo pages'
    )
  }

  // Query demo authors & slugs from search query in PennyLane Cloud
  const DemoAuthorResult = await graphql<Queries.GetDemoAuthorDataQuery>(`
    query GetDemoAuthorData {
      pennylaneCloud {
        search(input: { contentTypes: DEMO }) {
          items {
            ... on pennylaneCloud_GenericContent {
              authors {
                ... on pennylaneCloud_AuthorName {
                  name
                }
                ... on pennylaneCloud_Profile {
                  handle
                  firstName
                  headline
                  lastName
                  avatarUrl
                }
              }
              slug
            }
          }
        }
      }
    }
  `)

  if (DemoAuthorResult.errors) {
    reporter.panicOnBuild(
      `There was an error loading your demo authors`,
      DemoAuthorResult.errors
    )
    return
  }

  /**
   * Create a map for content slug and authors
   * to easily fetch authors for demo & blog pages while creating them programmatically.
   */
  const contentAuthorMap = {}
  const demoSearchItems =
    DemoAuthorResult.data?.pennylaneCloud.search.items || []

  demoSearchItems.forEach((content) => {
    contentAuthorMap[content['slug']] = content['authors']
  })

  // Create Pages Programmatically
  // -----------------------------
  // Create Demo Pages
  const demos = DemosResults.data
    ? DemosResults.data.allMarkdownRemark.nodes
    : []

  if (demos && demos.length) {
    demos.forEach((demo) => {
      if (demo.frontmatter?.slug) {
        createPage({
          path: `/qml/demos/${demo.frontmatter.slug}`,
          component: demoPage,
          context: {
            id: demo.id,
            authors: contentAuthorMap[demo.frontmatter.slug],
          },
        })
      }
    })
  }
}
